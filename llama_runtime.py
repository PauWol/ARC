# llama_runtime.py
"""
Managed llama-cpp-python runtime for small models (1.5–7 B) on low-end,
CPU-primary hardware.

Key design goals
----------------
- Lazy load / idle-timeout unload
- _load_lock (RLock)  is held only for short metadata operations
- _infer_lock (Lock)  serialises inference; held for the full call duration
  → management calls and the idle loop are never blocked by a running inference
- Full sync AND async generation APIs (chat, complete, streaming variants)
- RuntimePool for managing multiple models in one session
"""

from __future__ import annotations

import asyncio
import logging
import os
import time
import threading
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Any, AsyncIterator, Iterator, Sequence

try:
    import psutil  # optional — enables resource snapshots
except ImportError:
    psutil = None  # type: ignore

from llama_cpp import Llama, LlamaState

log = logging.getLogger(__name__)


@dataclass(slots=True)
class ModelSource:
    """
    Describes where the model lives.

    Use `model_path` for a local GGUF file, or `repo_id` + `filename` for
    automatic HuggingFace download via ``Llama.from_pretrained()``.
    """

    model_path: str | None = None
    repo_id: str | None = None
    filename: str | None = None
    additional_files: list[str] = field(default_factory=list)
    local_dir: str | None = None
    cache_dir: str | None = None

    def validate(self) -> None:
        local_ok = bool(self.model_path)
        hf_ok = bool(self.repo_id and self.filename)
        if local_ok == hf_ok:
            raise ValueError(
                "Provide either `model_path` OR (`repo_id` + `filename`), not both."
            )


@dataclass(slots=True)
class RuntimeOptions:
    """
    Runtime tuning for CPU-first, low-RAM hardware with optional GPU offload.

    Conservative defaults that behave well on 8 GB machines.
    Raise ``n_gpu_layers`` to offload; ``-1`` offloads all layers.

    Performance tips for low-end CPU
    ---------------------------------
    - ``n_batch=512``  speeds up prompt ingestion vs the old default of 256.
    - ``n_ubatch=128`` is a reasonable micro-batch; raise on machines with
      large L3 caches, lower on very tight RAM.
    - ``flash_attn=True`` can give a meaningful prefill speedup but requires a
      llama.cpp build compiled with ``LLAMA_FLASH_ATTN=1``.
    """

    n_ctx: int = 2048
    n_batch: int = 512  # was 256; 512 improves CPU prompt throughput
    n_ubatch: int = 128  # was 64; tune to your cache size
    n_threads: int | None = None
    n_threads_batch: int | None = None
    n_gpu_layers: int = 0
    split_mode: int | None = None
    main_gpu: int = 0
    seed: int = -1
    verbose: bool = False
    use_mlock: bool = False
    use_mmap: bool = True
    chat_format: str | None = None
    flash_attn: bool = False  # requires a flash-attn build of llama.cpp

    # Idle unload
    idle_unload_seconds: float | None = 300.0
    idle_check_interval: float = 5.0

    # Execution
    auto_load: bool = True
    enable_background_executor: bool = True

    # Sampling defaults (can be overridden per-call)
    temperature: float = 0.7
    top_p: float = 0.95
    top_k: int = 40
    repeat_penalty: float = 1.1

    def resolve_threads(self) -> tuple[int, int]:
        """Return (n_threads, n_threads_batch) with sensible CPU defaults."""
        cpu = max(os.cpu_count() or 1, 1)
        n_threads = self.n_threads if self.n_threads is not None else max(cpu // 2, 1)
        n_threads_batch = (
            self.n_threads_batch if self.n_threads_batch is not None else cpu
        )
        return n_threads, n_threads_batch


class LlamaRuntime:
    """
    Single-model managed runtime.

    Locking model
    -------------
    Two locks are used to avoid blocking management/idle calls during inference:

    ``_load_lock``  (RLock)
        Guards ``self._llm`` and other mutable fields.  Held only for brief
        metadata reads/writes, NOT during the actual inference call.

    ``_infer_lock``  (Lock)
        Serialises inference (llama.cpp is not safe for concurrent generation).
        Held for the full duration of a generation call.  The idle-unload loop
        acquires this before freeing the model so it never yanks the rug out
        from under a running inference.
    """

    def __init__(self, source: ModelSource, options: RuntimeOptions | None = None):
        self.source = source
        self.options = options or RuntimeOptions()
        self.source.validate()

        self._llm: Llama | None = None
        self._load_lock = threading.RLock()  # metadata guard (short sections)
        self._infer_lock = threading.Lock()  # inference serialiser (long sections)

        self._stop_event = threading.Event()
        self._last_used_monotonic = time.monotonic()
        self._last_usage: dict[str, int] | None = None
        self._saved_state: LlamaState | None = None

        self._executor: ThreadPoolExecutor | None = None
        if self.options.enable_background_executor:
            self._executor = ThreadPoolExecutor(
                max_workers=1, thread_name_prefix="llama-runtime"
            )

        self._idle_thread: threading.Thread | None = None
        if self.options.idle_unload_seconds is not None:
            self._idle_thread = threading.Thread(
                target=self._idle_loop,
                name="llama-idle-unloader",
                daemon=True,
            )
            self._idle_thread.start()

    # ── Lifecycle ─────────────────────────────────────────────────────────────

    @property
    def loaded(self) -> bool:
        """True when a model instance is currently held in memory."""
        return self._llm is not None

    @property
    def model(self) -> Llama:
        """Return the loaded Llama instance or raise if not loaded."""
        llm = self._llm
        if llm is None:
            raise RuntimeError("Model is not loaded.")
        return llm

    def load(self, force: bool = False) -> Llama:
        """
        Load the model if not already loaded, then return it.

        Thread-safe. Pass ``force=True`` to reload even if a model is present.
        """
        with self._load_lock:
            if self._llm is not None and not force:
                self._touch_locked()
                return self._llm
            if self._llm is not None and force:
                self._unload_locked()

            n_threads, n_threads_batch = self.options.resolve_threads()
            kwargs: dict[str, Any] = dict(
                n_ctx=self.options.n_ctx,
                n_batch=min(self.options.n_batch, self.options.n_ctx),
                n_ubatch=min(self.options.n_ubatch, self.options.n_batch),
                n_threads=n_threads,
                n_threads_batch=n_threads_batch,
                n_gpu_layers=self.options.n_gpu_layers,
                main_gpu=self.options.main_gpu,
                seed=self.options.seed,
                verbose=self.options.verbose,
                use_mlock=self.options.use_mlock,
                use_mmap=self.options.use_mmap,
                flash_attn=self.options.flash_attn,
            )
            if self.options.chat_format is not None:
                kwargs["chat_format"] = self.options.chat_format
            if self.options.split_mode is not None:
                kwargs["split_mode"] = self.options.split_mode

            log.info(
                "Loading model: %s",
                self.source.model_path or self.source.repo_id,
            )
            if self.source.model_path:
                self._llm = Llama(model_path=self.source.model_path, **kwargs)
            else:
                self._llm = Llama.from_pretrained(
                    repo_id=self.source.repo_id,  # type: ignore[arg-type]
                    filename=self.source.filename,  # type: ignore[arg-type]
                    additional_files=self.source.additional_files or None,
                    local_dir=self.source.local_dir,
                    cache_dir=self.source.cache_dir,
                    **kwargs,
                )
            log.info("Model loaded (n_ctx=%d).", self.options.n_ctx)
            self._touch_locked()
            return self._llm

    def unload(self) -> None:
        """
        Explicitly free the model from memory.

        Waits for any currently running inference to finish before freeing.
        """
        with self._infer_lock:  # don't pull the rug during inference
            with self._load_lock:
                self._unload_locked()

    def close(self) -> None:
        """
        Shut down all background helpers and unload the model.

        The executor is drained *first* so no new inference can start after
        the model is freed.
        """
        self._stop_event.set()
        if self._executor is not None:
            self._executor.shutdown(wait=True)  # drain in-flight work first
            self._executor = None
        with self._load_lock:
            self._unload_locked()

    def __enter__(self) -> "LlamaRuntime":
        if self.options.auto_load:
            self.load()
        return self

    def __exit__(self, *_: Any) -> None:
        self.close()

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:
            pass

    def _unload_locked(self) -> None:
        """Free model assuming ``_load_lock`` is already held."""
        if self._llm is not None:
            log.info("Unloading model.")
            try:
                self._llm.close()
            finally:
                self._llm = None
                self._saved_state = None
                self._last_usage = None

    def _touch_locked(self) -> None:
        """Refresh last-used timestamp; ``_load_lock`` must be held."""
        self._last_used_monotonic = time.monotonic()

    def touch(self) -> None:
        """Mark the runtime as recently used (resets idle timer)."""
        with self._load_lock:
            self._touch_locked()

    def ensure_loaded(self) -> Llama:
        """Load on demand and return the model instance."""
        if self._llm is None:
            return self.load()
        return self._llm

    # ── Context / token helpers ───────────────────────────────────────────────

    def context_window(self) -> int:
        """Return the model's configured context window size."""
        with self._load_lock:
            return self.ensure_loaded().n_ctx()

    def count_tokens(
        self, text: str, add_bos: bool = True, special: bool = False
    ) -> int:
        """Count tokens for a UTF-8 string using the model tokenizer."""
        with self._load_lock:
            return len(
                self.ensure_loaded().tokenize(text.encode("utf-8"), add_bos, special)
            )

    def current_context_tokens(self) -> int:
        """
        Return the exact number of tokens held in the active KV-cache state.

        Calls ``save_state()`` internally, which is not free — avoid in tight
        loops. Does NOT cache the state as a side effect; call
        ``save_context_state()`` explicitly if you need to checkpoint.
        """
        with self._load_lock:
            return int(self.ensure_loaded().save_state().n_tokens)

    def current_context_budget(self) -> dict[str, int]:
        """Return a compact snapshot of current KV usage and remaining capacity."""
        with self._load_lock:
            llm = self.ensure_loaded()  # single call — one save_state()
            n_ctx = llm.n_ctx()
            used = int(llm.save_state().n_tokens)
            return {
                "max_context_tokens": n_ctx,
                "used_context_tokens": used,
                "remaining_context_tokens": max(n_ctx - used, 0),
            }

    def token_budget_for_prompt(
        self, prompt: str, reserve_output_tokens: int = 256
    ) -> dict[str, int]:
        """Show prompt tokens, reserved output tokens, and remaining headroom."""
        with self._load_lock:
            max_ctx = self.ensure_loaded().n_ctx()
            prompt_tokens = self.count_tokens(prompt)
            remaining = max(max_ctx - prompt_tokens - reserve_output_tokens, 0)
            return {
                "max_context_tokens": max_ctx,
                "prompt_tokens": prompt_tokens,
                "reserved_output_tokens": reserve_output_tokens,
                "remaining_tokens": remaining,
            }

    # ── State management ──────────────────────────────────────────────────────

    def reset_context(self) -> None:
        """
        Clear the KV-cache without unloading weights.

        Fast path for starting a new conversation with a loaded model.
        """
        with self._load_lock:
            self.ensure_loaded().reset()
            self._last_usage = None
            self._saved_state = None
            self._touch_locked()

    def hard_clear(self) -> None:
        """
        Fully unload and reload the model.

        Use when ``reset_context()`` is not sufficient and you need a
        completely clean runtime state.
        """
        with self._load_lock:
            self._unload_locked()
            if self.options.auto_load:
                self.load()

    def save_context_state(self) -> LlamaState:
        """Checkpoint the current KV-cache state for later restore."""
        with self._load_lock:
            state = self.ensure_loaded().save_state()
            self._saved_state = state
            return state

    def restore_context_state(self, state: LlamaState | None = None) -> None:
        """
        Restore a previously saved KV-cache state.

        If no ``state`` is provided the last state saved by this runtime is used.
        """
        with self._load_lock:
            llm = self.ensure_loaded()
            chosen = state or self._saved_state
            if chosen is None:
                raise ValueError("No saved state available.")
            llm.load_state(chosen)
            self._touch_locked()

    # ── Internal generation helpers ───────────────────────────────────────────

    def _sampling_kwargs(
        self,
        temperature: float | None,
        top_p: float | None,
        top_k: int | None,
        repeat_penalty: float | None,
    ) -> dict[str, Any]:
        """Build sampling kwargs, falling back to RuntimeOptions defaults."""
        return {
            "temperature": self.options.temperature
            if temperature is None
            else temperature,
            "top_p": self.options.top_p if top_p is None else top_p,
            "top_k": self.options.top_k if top_k is None else top_k,
            "repeat_penalty": self.options.repeat_penalty
            if repeat_penalty is None
            else repeat_penalty,
        }

    def _record_usage(self, response: Any) -> None:
        if isinstance(response, dict):
            usage = response.get("usage") or {}
            self._last_usage = {
                "prompt_tokens": int(usage.get("prompt_tokens", 0)),
                "completion_tokens": int(usage.get("completion_tokens", 0)),
                "total_tokens": int(usage.get("total_tokens", 0)),
            }

    def _acquire_llm_for_inference(self, reset: bool) -> Llama:
        """
        Load the model (if needed), optionally reset context, touch timestamp.

        Called under ``_infer_lock`` before releasing ``_load_lock``.
        """
        with self._load_lock:
            llm = self.ensure_loaded()
            if reset:
                llm.reset()
                self._last_usage = None
                self._saved_state = None
            self._touch_locked()
        return llm

    # ── Sync generation ───────────────────────────────────────────────────────

    def complete(
        self,
        prompt: str,
        *,
        max_tokens: int = 256,
        stop: Sequence[str] | None = None,
        reset: bool = False,
        temperature: float | None = None,
        top_p: float | None = None,
        top_k: int | None = None,
        repeat_penalty: float | None = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """
        Blocking plain-text completion.

        ``_load_lock`` is released before inference begins so the idle loop
        and management calls remain responsive during long generations.
        """
        with self._infer_lock:
            llm = self._acquire_llm_for_inference(reset)
            # _load_lock is NOT held here
            response = llm.create_completion(
                prompt=prompt,
                max_tokens=max_tokens,
                stop=list(stop) if stop else None,
                stream=False,
                **self._sampling_kwargs(temperature, top_p, top_k, repeat_penalty),
                **kwargs,
            )
        self._record_usage(response)
        with self._load_lock:
            self._touch_locked()
        return response  # type: ignore[return-value]

    def stream_complete(
        self,
        prompt: str,
        *,
        max_tokens: int = 256,
        stop: Sequence[str] | None = None,
        reset: bool = False,
        temperature: float | None = None,
        top_p: float | None = None,
        top_k: int | None = None,
        repeat_penalty: float | None = None,
        **kwargs: Any,
    ) -> Iterator[Any]:
        """
        Streaming plain-text completion.

        Yields chunk dicts as they arrive.  ``_infer_lock`` is held for the
        full iteration so no other inference can interleave.
        """
        with self._infer_lock:
            llm = self._acquire_llm_for_inference(reset)
            yield from llm.create_completion(
                prompt=prompt,
                max_tokens=max_tokens,
                stop=list(stop) if stop else None,
                stream=True,
                **self._sampling_kwargs(temperature, top_p, top_k, repeat_penalty),
                **kwargs,
            )
        with self._load_lock:
            self._touch_locked()

    def chat(
        self,
        messages: Sequence[dict[str, str]],
        *,
        max_tokens: int = 256,
        stop: Sequence[str] | None = None,
        reset: bool = False,
        temperature: float | None = None,
        top_p: float | None = None,
        top_k: int | None = None,
        repeat_penalty: float | None = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """
        Blocking chat completion.

        Pass ``reset=True`` to clear the KV-cache before the call — useful
        when each chat() is a fresh conversation rather than a continuation.
        """
        with self._infer_lock:
            llm = self._acquire_llm_for_inference(reset)
            response = llm.create_chat_completion(
                messages=list(messages),  # type: ignore[arg-type]
                max_tokens=max_tokens,
                stop=list(stop) if stop else None,
                stream=False,
                **self._sampling_kwargs(temperature, top_p, top_k, repeat_penalty),
                **kwargs,
            )
        self._record_usage(response)
        with self._load_lock:
            self._touch_locked()
        return response  # type: ignore[return-value]

    def stream_chat(
        self,
        messages: Sequence[dict[str, str]],
        *,
        max_tokens: int = 256,
        stop: Sequence[str] | None = None,
        reset: bool = False,
        temperature: float | None = None,
        top_p: float | None = None,
        top_k: int | None = None,
        repeat_penalty: float | None = None,
        **kwargs: Any,
    ) -> Iterator[Any]:
        """
        Streaming chat completion.

        Yields server-sent-event-style chunk dicts.  Holds ``_infer_lock``
        for the full iteration.
        """
        with self._infer_lock:
            llm = self._acquire_llm_for_inference(reset)
            yield from llm.create_chat_completion(
                messages=list(messages),  # type: ignore[arg-type]
                max_tokens=max_tokens,
                stop=list(stop) if stop else None,
                stream=True,
                **self._sampling_kwargs(temperature, top_p, top_k, repeat_penalty),
                **kwargs,
            )
        with self._load_lock:
            self._touch_locked()

    # ── Async generation ──────────────────────────────────────────────────────
    #
    # All async methods run on the single-worker background executor so that
    # llama.cpp's non-thread-safe state is always touched from one thread.
    # Streaming methods bridge the sync generator to an async generator via
    # an asyncio.Queue, letting the caller ``async for`` over chunks without
    # blocking the event loop.

    def _require_executor(self) -> ThreadPoolExecutor:
        if self._executor is None:
            raise RuntimeError(
                "Background executor is disabled "
                "(set RuntimeOptions.enable_background_executor=True)."
            )
        return self._executor

    async def acomplete(self, *args: Any, **kwargs: Any) -> dict[str, Any]:
        """Async wrapper for ``complete()``."""
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(
            self._require_executor(),
            lambda: self.complete(*args, **kwargs),
        )

    async def achat(self, *args: Any, **kwargs: Any) -> dict[str, Any]:
        """Async wrapper for ``chat()``."""
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(
            self._require_executor(),
            lambda: self.chat(*args, **kwargs),
        )

    async def astream_complete(
        self,
        prompt: str,
        **kwargs: Any,
    ) -> AsyncIterator[Any]:
        """
        Async streaming completion.

        Usage::

            async for chunk in runtime.astream_complete("Once upon"):
                print(chunk["choices"][0]["text"], end="", flush=True)
        """
        executor = self._require_executor()
        loop = asyncio.get_running_loop()
        queue: asyncio.Queue[Any] = asyncio.Queue()
        _done = object()

        def _producer() -> None:
            try:
                for chunk in self.stream_complete(prompt, **kwargs):
                    loop.call_soon_threadsafe(queue.put_nowait, chunk)
            except Exception as exc:
                loop.call_soon_threadsafe(queue.put_nowait, exc)
            finally:
                loop.call_soon_threadsafe(queue.put_nowait, _done)

        executor.submit(_producer)
        while True:
            item = await queue.get()
            if item is _done:
                break
            if isinstance(item, BaseException):
                raise item
            yield item

    async def astream_chat(
        self,
        messages: Sequence[dict[str, str]],
        **kwargs: Any,
    ) -> AsyncIterator[Any]:
        """
        Async streaming chat.

        Usage::

            async for chunk in runtime.astream_chat(messages):
                delta = chunk["choices"][0]["delta"].get("content", "")
                print(delta, end="", flush=True)
        """
        executor = self._require_executor()
        loop = asyncio.get_running_loop()
        queue: asyncio.Queue[Any] = asyncio.Queue()
        _done = object()

        def _producer() -> None:
            try:
                for chunk in self.stream_chat(messages, **kwargs):
                    loop.call_soon_threadsafe(queue.put_nowait, chunk)
            except Exception as exc:
                loop.call_soon_threadsafe(queue.put_nowait, exc)
            finally:
                loop.call_soon_threadsafe(queue.put_nowait, _done)

        executor.submit(_producer)
        while True:
            item = await queue.get()
            if item is _done:
                break
            if isinstance(item, BaseException):
                raise item
            yield item

    # ── Background futures (sync callers) ─────────────────────────────────────

    def submit_complete(self, *args: Any, **kwargs: Any) -> Future:
        """Run ``complete()`` on the background executor; returns a Future."""
        return self._require_executor().submit(self.complete, *args, **kwargs)

    def submit_chat(self, *args: Any, **kwargs: Any) -> Future:
        """Run ``chat()`` on the background executor; returns a Future."""
        return self._require_executor().submit(self.chat, *args, **kwargs)

    @property
    def last_usage(self) -> dict[str, int] | None:
        """Token usage from the last non-streaming generation call."""
        return None if self._last_usage is None else dict(self._last_usage)

    # ── Resource / model info ─────────────────────────────────────────────────

    def model_info(self) -> dict[str, Any]:
        """
        Return a runtime snapshot.

        Avoids calling ``save_state()`` so this is cheap and does not alter
        internal state.
        """
        with self._load_lock:
            if not self.loaded:
                return {"loaded": False}
            llm = self._llm
            assert llm is not None
            info: dict[str, Any] = {
                "loaded": True,
                "context_window": llm.n_ctx(),
                "last_usage": dict(self._last_usage or {}),
            }
        if psutil is not None:
            proc = psutil.Process()
            info["rss_mb"] = round(proc.memory_info().rss / (1024 * 1024), 2)
            info["cpu_percent"] = proc.cpu_percent(interval=None)
        return info

    def resource_snapshot(self) -> dict[str, float]:
        """Return basic CPU / RAM stats (requires ``psutil``)."""
        if psutil is None:
            return {}
        proc = psutil.Process()
        return {
            "rss_mb": round(proc.memory_info().rss / (1024 * 1024), 2),
            "vms_mb": round(proc.memory_info().vms / (1024 * 1024), 2),
            "cpu_percent": proc.cpu_percent(interval=None),
        }

    # ── Idle unload ───────────────────────────────────────────────────────────

    def _idle_loop(self) -> None:
        """
        Background thread: unloads the model after ``idle_unload_seconds`` of
        inactivity.

        Uses a non-blocking ``_infer_lock`` attempt so the loop never stalls
        waiting for a long-running inference — it simply retries next interval.
        """
        interval = max(self.options.idle_check_interval, 0.5)
        timeout = self.options.idle_unload_seconds
        assert timeout is not None

        while not self._stop_event.wait(interval):
            # Quick metadata check (no inference contention)
            with self._load_lock:
                if self._llm is None:
                    continue
                idle_for = time.monotonic() - self._last_used_monotonic
                if idle_for < timeout:
                    continue

            # Model looks idle; try to grab infer_lock non-blocking so we
            # don't stall a just-started inference that reset the timestamp.
            if self._infer_lock.acquire(blocking=False):
                try:
                    with self._load_lock:
                        # Re-check: a new inference may have touched timestamp
                        idle_for = time.monotonic() - self._last_used_monotonic
                        if idle_for >= timeout:
                            log.info("Idle timeout reached — unloading model.")
                            self._unload_locked()
                finally:
                    self._infer_lock.release()


# ── RuntimePool ───────────────────────────────────────────────────────────────


class RuntimePool:
    """
    Manage a collection of named ``LlamaRuntime`` instances in one process.

    Typical use-case: multiple small models (a 1.5 B fast drafter and a 7 B
    reviewer) sharing the same session.  Models are loaded lazily and each
    retains its own idle-unload timer so they self-evict from RAM when unused.

    Example::

        pool = RuntimePool()
        pool.register(
            "drafter",
            ModelSource(model_path="./models/qwen1.5b.gguf"),
            RuntimeOptions(n_ctx=2048, n_gpu_layers=0, idle_unload_seconds=60),
        )
        pool.register(
            "reviewer",
            ModelSource(model_path="./models/qwen7b.gguf"),
            RuntimeOptions(n_ctx=4096, n_gpu_layers=20, idle_unload_seconds=120),
        )

        result = pool["drafter"].chat(messages)
        result = await pool["reviewer"].achat(messages)

        pool.close_all()
    """

    def __init__(self) -> None:
        self._runtimes: dict[str, LlamaRuntime] = {}
        self._lock = threading.Lock()

    def register(
        self,
        name: str,
        source: ModelSource,
        options: RuntimeOptions | None = None,
        *,
        replace: bool = False,
    ) -> LlamaRuntime:
        """
        Create and register a runtime under ``name``.

        Raises ``KeyError`` if the name already exists unless ``replace=True``.
        """
        with self._lock:
            if name in self._runtimes and not replace:
                raise KeyError(
                    f"A runtime named {name!r} already exists. "
                    "Pass replace=True to overwrite."
                )
            if name in self._runtimes and replace:
                self._runtimes[name].close()
            rt = LlamaRuntime(source, options)
            self._runtimes[name] = rt
            log.info("RuntimePool: registered %r.", name)
            return rt

    def get(self, name: str) -> LlamaRuntime:
        """Return the runtime for ``name`` or raise ``KeyError``."""
        try:
            return self._runtimes[name]
        except KeyError:
            raise KeyError(f"No runtime named {name!r} in pool.") from None

    def __getitem__(self, name: str) -> LlamaRuntime:
        return self.get(name)

    def names(self) -> list[str]:
        """Return a list of registered runtime names."""
        with self._lock:
            return list(self._runtimes)

    def loaded_names(self) -> list[str]:
        """Return the names of runtimes that currently have a model in memory."""
        with self._lock:
            return [n for n, rt in self._runtimes.items() if rt.loaded]

    def unload_all(self) -> None:
        """Unload all models without destroying the runtimes (they can reload)."""
        with self._lock:
            for rt in self._runtimes.values():
                rt.unload()

    def close_all(self) -> None:
        """Shut down and free every registered runtime."""
        with self._lock:
            for name, rt in self._runtimes.items():
                log.info("RuntimePool: closing %r.", name)
                rt.close()
            self._runtimes.clear()

    def resource_snapshot(self) -> dict[str, Any]:
        """Return per-runtime load status plus an optional process-level RSS."""
        snapshot: dict[str, Any] = {
            n: {"loaded": rt.loaded, "last_usage": rt.last_usage}
            for n, rt in self._runtimes.items()
        }
        if psutil is not None:
            proc = psutil.Process()
            snapshot["_process"] = {
                "rss_mb": round(proc.memory_info().rss / (1024 * 1024), 2),
            }
        return snapshot

    def __enter__(self) -> "RuntimePool":
        return self

    def __exit__(self, *_: Any) -> None:
        self.close_all()
