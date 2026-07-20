"""
Managed llama-cpp-python runtime.

What's new vs v1
----------------
HardwareProfile   Detects GPU support, physical cores, and RAM once at startup.
                  Every default is derived from it — you rarely need to tune.

RuntimeOptions    auto() / fast() / cpu_only() factory presets.
                  build_kwargs() resolves all "auto" fields internally.

GenerationConfig  Typed, reusable sampling params. Keep presets as constants
                  and pass them per-call instead of scattering kwargs everywhere.

LlamaRuntime      set_system_prompt() pre-fills the KV cache with a system
                  prompt so every subsequent chat() only processes new tokens.
                  from_path() / from_hf() constructors replace the raw __init__.
                  chat() / complete() accept an optional config= argument.

                  chat()/stream_chat() take tools= / tool_choice= /
                  response_format= directly (forwarded to
                  create_chat_completion, not through GenerationConfig — tool
                  schemas relies on the model's own chat template; leave
                  RuntimeOptions.chat_format=None to auto-detect it from the
                  GGUF, or set chat_format="chatml-function-calling" as a
                  fallback for models with no native tool-calling template).

                  reset now defaults to False on chat()/stream_chat(): each
                  call is assumed to be one turn of an ongoing conversation,
                  so llama.cpp's own KV prefix-cache reuse does the work
                  instead of reprocessing the whole history every turn. Pass
                  reset=True yourself only for a conversation's first turn —
                  or just use Conversation, below, which tracks that for you.

Conversation      Wraps a LlamaRuntime for a multi-turn agent loop: tracks
                  message history, calls chat() with reset=True only on the
                  first turn, and gives you send() / send_tool_results().

RuntimePool       Direct dispatch: pool.chat(name, msgs) / pool.complete(name, prompt).
                  LRU eviction: when max_loaded is set the least-recently-used
                  model is unloaded automatically before loading a new one.
"""

from __future__ import annotations

import asyncio
import hashlib
import logging
import os
import time
import threading
import psutil
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass, field, replace
from typing import Any
from collections.abc import AsyncIterator, Iterator, Sequence

from llama_cpp import Llama, LlamaState

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class HardwareProfile:
    """
    Immutable snapshot of hardware capabilities detected once at startup.

    Usage::

        hw = HardwareProfile.detect()
        print(hw)  # "8p/16l cores  32.0 GB RAM  GPU"
    """

    physical_cores: int
    logical_cores: int
    total_ram_gb: float
    free_ram_gb: float
    gpu_offload: bool  # True if the llama.cpp build supports GPU offload

    @classmethod
    def detect(cls) -> "HardwareProfile":
        """Auto-detect hardware — cheap, side-effect-free."""
        logical = os.cpu_count() or 1
        physical = logical
        total_ram = free_ram = 0.0

        physical = psutil.cpu_count(logical=False) or logical
        mem = psutil.virtual_memory()
        total_ram = mem.total / (1024**3)
        free_ram = mem.available / (1024**3)

        try:
            from llama_cpp import llama_supports_gpu_offload  # type: ignore

            gpu = llama_supports_gpu_offload()
        except ImportError, AttributeError:
            gpu = False

        profile = cls(
            physical_cores=physical,
            logical_cores=logical,
            total_ram_gb=round(total_ram, 1),
            free_ram_gb=round(free_ram, 1),
            gpu_offload=gpu,
        )
        log.info("Hardware: %s", profile)
        return profile

    def __str__(self) -> str:
        tag = "GPU" if self.gpu_offload else "CPU-only"
        return (
            f"{self.physical_cores}p/{self.logical_cores}l cores  "
            f"{self.total_ram_gb:.0f} GB RAM  {tag}"
        )


# Module-level hardware cache — detected once, reused everywhere.
_HW: HardwareProfile | None = None


def hardware() -> HardwareProfile:
    """Return the cached HardwareProfile, detecting it on first call."""
    global _HW
    if _HW is None:
        _HW = HardwareProfile.detect()
    return _HW


@dataclass
class GenerationConfig:
    """
    Per-call sampling parameters.

    Keep module-level presets and pass them to chat() / complete() via
    the ``config=`` argument instead of repeating kwargs everywhere::

        PLAN_CFG = GenerationConfig(max_tokens=120, temperature=0.0, top_p=0.1)
        runtime.chat(messages, config=PLAN_CFG)

    Override individual fields with .merge()::

        long_cfg = PLAN_CFG.merge(max_tokens=512)
    """

    max_tokens: int = 256
    temperature: float = 0.7
    top_p: float = 0.95
    top_k: int = 40
    repeat_penalty: float = 1.1
    stop: list[str] = field(default_factory=list)
    grammar: Any = None  # LlamaGrammar | None

    def merge(self, **overrides: Any) -> "GenerationConfig":
        """Return a new config with the given fields overridden."""
        return replace(self, **overrides)

    def to_kwargs(self) -> dict[str, Any]:
        """Convert to llama-cpp-python generation kwargs."""
        kw: dict[str, Any] = {
            "max_tokens": self.max_tokens,
            "temperature": self.temperature,
            "top_p": self.top_p,
            "top_k": self.top_k,
            "repeat_penalty": self.repeat_penalty,
        }
        if self.stop:
            kw["stop"] = self.stop
        if self.grammar is not None:
            kw["grammar"] = self.grammar
        return kw


# Convenience presets — import and use directly.
GEN_DEFAULT = GenerationConfig()
GEN_GREEDY = GenerationConfig(temperature=0.0, top_p=1.0, top_k=1)
GEN_FAST = GenerationConfig(max_tokens=128, temperature=0.0, top_p=0.1)


@dataclass(slots=True)
class ModelSource:
    """
    Describes where to load the model from.

    Provide either ``model_path`` (local GGUF) or ``repo_id`` + ``filename``
    (automatic HuggingFace download via ``Llama.from_pretrained``).
    """

    model_path: str | None = None
    repo_id: str | None = None
    filename: str | None = None
    additional_files: list[str] = field(default_factory=list)
    local_dir: str | None = None
    cache_dir: str | None = None
    # Manual override for model-family auto-detection (see src.models.profiles).
    # Leave as None to let detect_profile() fingerprint path/repo_id/filename.
    family: str | None = None

    @property
    def display_name(self) -> str:
        return self.model_path or f"{self.repo_id}/{self.filename}" or "unknown"

    def validate(self) -> None:
        local_ok = bool(self.model_path)
        hf_ok = bool(self.repo_id and self.filename)
        if local_ok == hf_ok:
            raise ValueError(
                "Provide either model_path OR (repo_id + filename), not both/neither."
            )


@dataclass
class RuntimeOptions:
    """
    Model loading and execution configuration.

    All values default to "auto" (None) where hardware detection applies.
    Prefer the factory classmethods::

        RuntimeOptions.auto()                # best for current hardware
        RuntimeOptions.fast()                # maximize throughput
        RuntimeOptions.cpu_only(n_ctx=512)   # force CPU, specific context
        RuntimeOptions.auto().with_ctx(512)  # auto + small context window
    """

    # Context
    n_ctx: int = 2048
    n_batch: int = 512
    n_ubatch: int = 128

    # Threads — None resolves via HardwareProfile at load time
    n_threads: int | None = None
    n_threads_batch: int | None = None

    # GPU — None = auto (-1 if GPU build, 0 if not); -1 = offload all layers
    n_gpu_layers: int | None = None
    main_gpu: int = 0
    split_mode: int | None = None

    # Features
    flash_attn: bool = True  # safe with any build; silently ignored if not compiled in
    use_mmap: bool = True
    use_mlock: bool = False
    seed: int = -1
    verbose: bool = False
    chat_format: str | None = None

    # Idle management
    idle_unload_seconds: float | None = 300.0
    idle_check_interval: float = 5.0

    # Behavior
    auto_load: bool = True
    enable_background_executor: bool = True

    # Default sampling — used when no GenerationConfig is passed to chat/complete
    default_generation: GenerationConfig = field(default_factory=GenerationConfig)

    @classmethod
    def auto(cls, **overrides: Any) -> "RuntimeOptions":
        """Best settings for detected hardware: GPU if available, physical core threads."""
        hw = hardware()
        return cls(n_gpu_layers=-1 if hw.gpu_offload else 0, **overrides)

    @classmethod
    def fast(cls, **overrides: Any) -> "RuntimeOptions":
        """Maximum throughput: full GPU offload, large batches, mlock."""
        return cls(
            n_gpu_layers=-1,
            flash_attn=True,
            n_batch=1024,
            n_ubatch=512,
            use_mlock=True,
            idle_unload_seconds=None,  # never evict a fast model
            **overrides,
        )

    @classmethod
    def cpu_only(cls, **overrides: Any) -> "RuntimeOptions":
        """Force CPU execution regardless of GPU availability."""
        return cls(n_gpu_layers=0, flash_attn=False, **overrides)

    def with_ctx(self, n_ctx: int) -> "RuntimeOptions":
        """Return a copy with a specific context window size."""
        return replace(self, n_ctx=n_ctx)

    def with_overrides(self, **overrides: Any) -> "RuntimeOptions":
        """Return a copy with arbitrary fields overridden."""
        return replace(self, **overrides)

    def build_kwargs(self, hw: HardwareProfile | None = None) -> dict[str, Any]:
        """
        Resolve all auto fields and return the kwargs dict for Llama().

        Called once inside load(); not normally needed by callers.
        """
        hw = hw or hardware()

        n_threads = self.n_threads if self.n_threads is not None else hw.physical_cores
        n_threads_batch = (
            self.n_threads_batch
            if self.n_threads_batch is not None
            else hw.logical_cores
        )

        # Resolve n_gpu_layers
        if self.n_gpu_layers is None:
            gpu_layers = -1 if hw.gpu_offload else 0
        else:
            gpu_layers = self.n_gpu_layers
        if not hw.gpu_offload and gpu_layers != 0:
            log.debug("GPU not available in this build — forcing n_gpu_layers=0.")
            gpu_layers = 0

        kwargs: dict[str, Any] = dict(
            n_ctx=self.n_ctx,
            n_batch=min(self.n_batch, self.n_ctx),
            n_ubatch=min(self.n_ubatch, self.n_batch),
            n_threads=n_threads,
            n_threads_batch=n_threads_batch,
            n_gpu_layers=gpu_layers,
            main_gpu=self.main_gpu,
            seed=self.seed,
            verbose=self.verbose,
            use_mlock=self.use_mlock,
            use_mmap=self.use_mmap,
            flash_attn=self.flash_attn,
        )
        if self.chat_format is not None:
            kwargs["chat_format"] = self.chat_format
        if self.split_mode is not None:
            kwargs["split_mode"] = self.split_mode
        return kwargs


class LlamaRuntime:
    """
    Single-model managed runtime.

    Locking model
    -------------
    _load_lock  (RLock)  Guards _llm and mutable metadata.  Held only during
                         brief reads/writes — never during inference.
    _infer_lock (Lock)   Serialises all generation calls. Held for the full
                         duration of each inference. The idle-unload loop uses
                         a non-blocking acquire so it never stalls a running call.

    System-prompt KV caching
    -------------------------
    Call set_system_prompt(text) once after loading the model.  Every
    subsequent chat() call that starts with the same system message will
    restore from the saved KV state and only process the new user tokens,
    saving ~200-400 ms per step on CPU.
    """

    def __init__(self, source: ModelSource, options: RuntimeOptions | None = None):
        self.source = source
        self.options = options or RuntimeOptions.auto()
        self.source.validate()

        self._llm: Llama | None = None
        self._load_lock = threading.RLock()
        self._infer_lock = threading.Lock()

        self._stop_event = threading.Event()
        self._last_used_monotonic = time.monotonic()
        self._last_usage: dict[str, int] | None = None
        self._saved_state: LlamaState | None = None

        # System-prompt KV cache
        self._sys_cache_key: str | None = None
        self._sys_cache_content: str | None = None
        self._sys_cache_state: LlamaState | None = None

        self._executor: ThreadPoolExecutor | None = None
        if self.options.enable_background_executor:
            self._executor = ThreadPoolExecutor(
                max_workers=1, thread_name_prefix="llama-runtime"
            )

        self._idle_thread: threading.Thread | None = None
        if self.options.idle_unload_seconds is not None:
            self._idle_thread = threading.Thread(
                target=self._idle_loop,
                name="llama-idle",
                daemon=True,
            )
            self._idle_thread.start()

        self._profile = None  # lazily resolved, see .profile

    @property
    def profile(self):
        """
        Model-family profile (thinking/non-thinking, sampling defaults,
        system-role support, grammar safety). Resolved once and cached;
        set self.source.family to override auto-detection.
        """
        if self._profile is None:
            from src.agent.profiles import detect_profile

            self._profile = detect_profile(self.source, override=self.source.family)
        return self._profile

    # ── Constructors ──────────────────────────────────────────────────────────

    @classmethod
    def from_path(
        cls,
        model_path: str,
        options: RuntimeOptions | None = None,
    ) -> "LlamaRuntime":
        """Create a runtime from a local GGUF file with auto-detected options."""
        return cls(ModelSource(model_path=model_path), options or RuntimeOptions.auto())

    @classmethod
    def from_hf(
        cls,
        repo_id: str,
        filename: str,
        options: RuntimeOptions | None = None,
        *,
        cache_dir: str | None = None,
        local_dir: str | None = None,
    ) -> "LlamaRuntime":
        """Create a runtime from a HuggingFace repo (downloads on first use)."""
        src = ModelSource(
            repo_id=repo_id,
            filename=filename,
            cache_dir=cache_dir,
            local_dir=local_dir,
        )
        return cls(src, options or RuntimeOptions.auto())

    @property
    def loaded(self) -> bool:
        return self._llm is not None

    @property
    def model(self) -> Llama:
        llm = self._llm
        if llm is None:
            raise RuntimeError("Model is not loaded.")
        return llm

    def load(self, force: bool = False) -> Llama:
        """Load the model if not already loaded; return the Llama instance."""
        with self._load_lock:
            if self._llm is not None and not force:
                self._touch()
                return self._llm
            if self._llm is not None:
                self._unload()

            kwargs = self.options.build_kwargs()
            log.info("Loading %s", self.source.display_name)

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
            log.info("Loaded (n_ctx=%d).", self.options.n_ctx)
            self._touch()
            return self._llm

    def unload(self) -> None:
        """Free the model from memory (waits for any running inference)."""
        with self._infer_lock:
            with self._load_lock:
                self._unload()

    def close(self) -> None:
        """Drain the background executor, then unload the model."""
        self._stop_event.set()
        if self._executor is not None:
            self._executor.shutdown(wait=True)
            self._executor = None
        with self._load_lock:
            self._unload()

    def ensure_loaded(self) -> Llama:
        """Load on demand and return the model."""
        return self._llm if self._llm is not None else self.load()

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

    def set_system_prompt(self, content: str) -> None:
        """
        Pre-fill the KV cache with a system prompt and save the state.

        After this call, every chat() whose first message is this system
        prompt will restore from the saved state and skip re-processing the
        prefix tokens — a significant saving on CPU-bound deployments.

        Call once after load(); call again if the prompt changes.
        """
        key = hashlib.sha256(content.encode()).hexdigest()
        with self._load_lock:
            if key == self._sys_cache_key:
                return  # already cached — nothing to do

        with self._infer_lock:
            llm = self._prepare_inference(reset=True)
            # warm the KV cache; generate 1 token to force a full forward pass
            llm.create_chat_completion(
                messages=[{"role": "system", "content": content}],
                max_tokens=1,
                temperature=0.0,
            )
            state = llm.save_state()
            with self._load_lock:
                self._sys_cache_key = key
                self._sys_cache_content = content
                self._sys_cache_state = state

        log.debug(
            "System prompt cached (%d chars, %d tokens in KV).",
            len(content),
            state.n_tokens,
        )

    def clear_system_prompt(self) -> None:
        """Discard the cached system prompt state."""
        with self._load_lock:
            self._sys_cache_key = None
            self._sys_cache_content = None
            self._sys_cache_state = None

    def context_window(self) -> int:
        """Return the context window size."""
        with self._load_lock:
            return self.ensure_loaded().n_ctx()

    def count_tokens(self, text: str, add_bos: bool = True) -> int:
        """Count the amount of tokens that will be used by input :param: text."""
        with self._load_lock:
            return len(self.ensure_loaded().tokenize(text.encode(), add_bos))

    def context_budget(self) -> dict[str, int]:
        """
        Snapshot of current KV usage and remaining capacity.

        Uses the cheap ``Llama.n_tokens`` counter that llama-cpp-python
        already tracks internally. Falls back to save_state() (which copies
        the entire KV cache just to read one integer) only on versions where
        that attribute isn't exposed.
        """
        with self._load_lock:
            llm = self.ensure_loaded()
            n_ctx = llm.n_ctx()
            used = getattr(llm, "n_tokens", None)
            if used is None:
                log.debug(
                    "llama_cpp.Llama has no n_tokens attribute on this version; "
                    "falling back to save_state() for context_budget() (expensive)."
                )
                used = llm.save_state().n_tokens
            used = int(used)
            return {
                "max": n_ctx,
                "used": used,
                "remaining": max(n_ctx - used, 0),
            }

    def reset_context(self) -> None:
        """Clear the KV-cache without unloading weights."""
        with self._load_lock:
            self.ensure_loaded().reset()
            self._last_usage = None
            self._saved_state = None
            self._touch()

    def save_context_state(self) -> LlamaState:
        with self._load_lock:
            state = self.ensure_loaded().save_state()
            self._saved_state = state
            return state

    def restore_context_state(self, state: LlamaState | None = None) -> None:
        with self._load_lock:
            chosen = state or self._saved_state
            if chosen is None:
                raise ValueError("No saved state available.")
            self.ensure_loaded().load_state(chosen)
            self._touch()

    def complete(
        self,
        prompt: str,
        *,
        config: GenerationConfig | None = None,
        reset: bool = False,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """
        Blocking plain-text completion.

        ``config`` sets sampling params.  Any extra ``kwargs`` are forwarded
        directly to ``create_completion`` as overrides.
        """
        cfg = self._resolve_config(config, **kwargs)
        with self._infer_lock:
            llm = self._prepare_inference(reset=reset)
            response = llm.create_completion(
                prompt=prompt, stream=False, **cfg.to_kwargs()
            )
        self._record_usage(response)
        self._touch()
        return response  # type: ignore[return-value]

    def stream_complete(
        self,
        prompt: str,
        *,
        config: GenerationConfig | None = None,
        reset: bool = False,
        **kwargs: Any,
    ) -> Iterator[Any]:
        """Streaming plain-text completion; yields chunk dicts."""
        cfg = self._resolve_config(config, **kwargs)
        with self._infer_lock:
            llm = self._prepare_inference(reset=reset)
            yield from llm.create_completion(
                prompt=prompt, stream=True, **cfg.to_kwargs()
            )
        self._touch()

    def chat(
        self,
        messages: Sequence[dict[str, str]],
        *,
        config: GenerationConfig | None = None,
        reset: bool = False,
        tools: list[dict[str, Any]] | None = None,
        tool_choice: str | dict[str, Any] | None = None,
        response_format: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """
        Blocking chat completion.

        ``tools`` / ``tool_choice`` / ``response_format`` are forwarded
        straight to ``create_chat_completion`` — kept out of GenerationConfig
        entirely so they can never collide with sampling-param merging.
        They rely on the loaded GGUF's own tool-calling chat template being
        auto-detected (leave ``RuntimeOptions.chat_format=None``), or on an
        explicit fallback like ``chat_format="chatml-function-calling"`` for
        models with no native tool template.

        ``reset`` defaults to False: this call assumes it is one turn in an
        ongoing conversation, and lets llama.cpp's own longest-common-prefix
        KV cache reuse do its job instead of reprocessing the whole history
        every turn. Pass ``reset=True`` only for the first turn of a new
        conversation (or use the ``Conversation`` helper, which does this
        for you automatically).

        If a system prompt was cached via set_system_prompt() and the first
        message matches it, the KV state is restored so that prefix isn't
        re-processed either.
        """
        cfg = self._resolve_config(config, **kwargs)
        call_kwargs = self._tool_kwargs(tools, tool_choice, response_format)
        with self._infer_lock:
            sys_state = self._matching_sys_state(messages)
            if sys_state is not None:
                llm = self._prepare_inference(reset=True)
                llm.load_state(sys_state)
            else:
                llm = self._prepare_inference(reset=reset)
            response = llm.create_chat_completion(
                messages=list(messages),  # type: ignore[arg-type]
                stream=False,
                **cfg.to_kwargs(),
                **call_kwargs,
            )
        self._record_usage(response)
        self._touch()
        return response  # type: ignore[return-value]

    def stream_chat(
        self,
        messages: Sequence[dict[str, str]],
        *,
        config: GenerationConfig | None = None,
        reset: bool = False,
        tools: list[dict[str, Any]] | None = None,
        tool_choice: str | dict[str, Any] | None = None,
        response_format: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> Iterator[Any]:
        """Streaming chat completion; yields chunk dicts. See chat() for
        the meaning of ``reset`` and the tool-calling parameters."""
        cfg = self._resolve_config(config, **kwargs)
        call_kwargs = self._tool_kwargs(tools, tool_choice, response_format)
        with self._infer_lock:
            sys_state = self._matching_sys_state(messages)
            if sys_state is not None:
                llm = self._prepare_inference(reset=True)
                llm.load_state(sys_state)
            else:
                llm = self._prepare_inference(reset=reset)
            yield from llm.create_chat_completion(
                messages=list(messages),  # type: ignore[arg-type]
                stream=True,
                **cfg.to_kwargs(),
                **call_kwargs,
            )
        self._touch()

    @staticmethod
    def _tool_kwargs(
        tools: list[dict[str, Any]] | None,
        tool_choice: str | dict[str, Any] | None,
        response_format: dict[str, Any] | None,
    ) -> dict[str, Any]:
        """Build the extra create_chat_completion kwargs for tool calling.

        Kept separate from GenerationConfig on purpose: GenerationConfig is a
        fixed-field dataclass (merge() uses dataclasses.replace), so passing
        tools=... through it would raise a TypeError instead of doing
        anything useful.
        """
        kw: dict[str, Any] = {}
        if tools is not None:
            kw["tools"] = tools
        if tool_choice is not None:
            kw["tool_choice"] = tool_choice
        if response_format is not None:
            kw["response_format"] = response_format
        return kw

    async def acomplete(self, *args: Any, **kwargs: Any) -> dict[str, Any]:
        """Async wrapper for complete()."""
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(
            self._require_executor(), lambda: self.complete(*args, **kwargs)
        )

    async def achat(self, *args: Any, **kwargs: Any) -> dict[str, Any]:
        """Async wrapper for chat()."""
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(
            self._require_executor(), lambda: self.chat(*args, **kwargs)
        )

    async def astream_complete(self, prompt: str, **kwargs: Any) -> AsyncIterator[Any]:
        """Async streaming completion."""
        async for chunk in self._bridge_stream(self.stream_complete, prompt, **kwargs):
            yield chunk

    async def astream_chat(
        self, messages: Sequence[dict[str, str]], **kwargs: Any
    ) -> AsyncIterator[Any]:
        """Async streaming chat."""
        async for chunk in self._bridge_stream(self.stream_chat, messages, **kwargs):
            yield chunk

    def submit_complete(self, *args: Any, **kwargs: Any) -> Future:
        return self._require_executor().submit(self.complete, *args, **kwargs)

    def submit_chat(self, *args: Any, **kwargs: Any) -> Future:
        return self._require_executor().submit(self.chat, *args, **kwargs)

    @property
    def last_usage(self) -> dict[str, int] | None:
        return None if self._last_usage is None else dict(self._last_usage)

    def model_info(self) -> dict[str, Any]:
        """Cheap runtime snapshot — does not call save_state()."""
        with self._load_lock:
            if not self.loaded:
                return {"loaded": False}
            info: dict[str, Any] = {
                "loaded": True,
                "model": self.source.display_name,
                "context_window": self._llm.n_ctx(),  # type: ignore[union-attr]
                "last_usage": dict(self._last_usage or {}),
                "sys_cache": self._sys_cache_key is not None,
            }
        if psutil is not None:
            proc = psutil.Process()
            info["rss_mb"] = round(proc.memory_info().rss / (1024**2), 1)
            info["cpu_percent"] = proc.cpu_percent(interval=None)
        return info

    def _prepare_inference(self, reset: bool) -> Llama:
        """Load model on demand, optionally reset context, refresh timestamp."""
        with self._load_lock:
            llm = self.ensure_loaded()
            if reset:
                llm.reset()
                self._last_usage = None
                self._saved_state = None
            self._touch()
        return llm

    def _matching_sys_state(
        self, messages: Sequence[dict[str, str]]
    ) -> LlamaState | None:
        """Return the cached KV state if the first message matches it."""
        if not messages:
            return None
        with self._load_lock:
            if self._sys_cache_state is None:
                return None
            first = messages[0]
            if (
                first.get("role") == "system"
                and first.get("content") == self._sys_cache_content
            ):
                return self._sys_cache_state
        return None

    def _resolve_config(
        self, config: GenerationConfig | None, **kwargs: Any
    ) -> GenerationConfig:
        """
        Merge call-site config with runtime defaults.

        Priority: explicit kwargs > config argument > options.default_generation.
        """
        base = config or self.options.default_generation
        return base.merge(**kwargs) if kwargs else base

    def _record_usage(self, response: Any) -> None:
        if isinstance(response, dict):
            usage = response.get("usage") or {}
            self._last_usage = {
                "prompt_tokens": int(usage.get("prompt_tokens", 0)),
                "completion_tokens": int(usage.get("completion_tokens", 0)),
                "total_tokens": int(usage.get("total_tokens", 0)),
            }

    def _touch(self) -> None:
        """Refresh the last-used timestamp (call with _load_lock held or not)."""
        self._last_used_monotonic = time.monotonic()

    def _unload(self) -> None:
        """Free the model; caller must hold _load_lock."""
        if self._llm is not None:
            log.info("Unloading %s.", self.source.display_name)
            try:
                self._llm.close()
            finally:
                self._llm = None
                self._saved_state = None
                self._last_usage = None
                self._sys_cache_state = None  # state is tied to a loaded model

    def _require_executor(self) -> ThreadPoolExecutor:
        if self._executor is None:
            raise RuntimeError(
                "Background executor disabled. "
                "Set RuntimeOptions.enable_background_executor=True."
            )
        return self._executor

    async def _bridge_stream(
        self, fn: Any, *args: Any, **kwargs: Any
    ) -> AsyncIterator[Any]:
        """Bridge a sync generator to an async generator via a queue."""
        executor = self._require_executor()
        loop = asyncio.get_running_loop()
        queue: asyncio.Queue[Any] = asyncio.Queue()
        _done = object()

        def _produce() -> None:
            try:
                for chunk in fn(*args, **kwargs):
                    loop.call_soon_threadsafe(queue.put_nowait, chunk)
            except Exception as exc:
                loop.call_soon_threadsafe(queue.put_nowait, exc)
            finally:
                loop.call_soon_threadsafe(queue.put_nowait, _done)

        executor.submit(_produce)
        while True:
            item = await queue.get()
            if item is _done:
                break
            if isinstance(item, BaseException):
                raise item
            yield item

    def _idle_loop(self) -> None:
        interval = max(self.options.idle_check_interval, 0.5)
        timeout = self.options.idle_unload_seconds
        assert timeout is not None

        while not self._stop_event.wait(interval):
            with self._load_lock:
                if self._llm is None:
                    continue
                idle = time.monotonic() - self._last_used_monotonic
                if idle < timeout:
                    continue
            if self._infer_lock.acquire(blocking=False):
                try:
                    with self._load_lock:
                        if time.monotonic() - self._last_used_monotonic >= timeout:
                            log.info(
                                "Idle timeout — unloading %s.", self.source.display_name
                            )
                            self._unload()
                finally:
                    self._infer_lock.release()


class Conversation:
    """
    Tracks one ongoing multi-turn chat against a LlamaRuntime.

    This exists to make the reset=True-only-on-the-first-turn pattern hard
    to get wrong. Calling ``runtime.chat()`` directly with the wrong reset
    value is exactly what silently defeats llama.cpp's own KV prefix reuse
    (reset=True every turn) or leaks a previous session's context into a new
    one (reset=False on the first turn) — Conversation just tracks which
    turn it is so you don't have to.

    Usage::

        convo = Conversation(runtime, system="You are a helpful assistant.")
        reply = convo.send("What's the weather in Berlin?", tools=tools)
        # ... execute any tool_calls in reply, then feed results back:
        reply = convo.send_tool_results(tool_results)

    Not thread-safe across concurrent send() calls on the same Conversation
    (the underlying LlamaRuntime's _infer_lock serializes actual inference,
    but message-history bookkeeping here is not itself locked).
    """

    def __init__(self, runtime: "LlamaRuntime", system: str | None = None) -> None:
        self.runtime = runtime
        self.messages: list[dict[str, Any]] = []
        self._started = False
        if system is not None:
            self.messages.append({"role": "system", "content": system})

    def send(
        self,
        content: str,
        *,
        config: GenerationConfig | None = None,
        tools: list[dict[str, Any]] | None = None,
        tool_choice: str | dict[str, Any] | None = None,
        response_format: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Append a user message, run one turn, append the assistant reply."""
        self.messages.append({"role": "user", "content": content})
        return self._run(
            config=config,
            tools=tools,
            tool_choice=tool_choice,
            response_format=response_format,
            **kwargs,
        )

    def send_tool_results(
        self,
        results: Sequence[dict[str, Any]],
        *,
        config: GenerationConfig | None = None,
        tools: list[dict[str, Any]] | None = None,
        tool_choice: str | dict[str, Any] | None = None,
        response_format: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """
        Append one or more tool-result messages and run the next turn.

        Each item in ``results`` should already be a proper message dict,
        e.g. ``{"role": "tool", "tool_call_id": ..., "content": ...}``.
        """
        self.messages.extend(results)
        return self._run(
            config=config,
            tools=tools,
            tool_choice=tool_choice,
            response_format=response_format,
            **kwargs,
        )

    def _run(self, **kwargs: Any) -> dict[str, Any]:
        reset = not self._started
        response = self.runtime.chat(self.messages, reset=reset, **kwargs)
        self._started = True
        choice = response["choices"][0]["message"]
        self.messages.append(dict(choice))
        return response

    def reset(self, system: str | None = None) -> None:
        """Start a fresh conversation (next send() will reset the KV cache)."""
        self.messages = []
        self._started = False
        if system is not None:
            self.messages.append({"role": "system", "content": system})

    def send_stream(
        self,
        content: str,
        *,
        config: GenerationConfig | None = None,
        tools: list[dict[str, Any]] | None = None,
        tool_choice: str | dict[str, Any] | None = None,
        response_format: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> Iterator[dict[str, Any]]:
        """
        Streaming counterpart to send().

        Yields raw chunk dicts as they arrive (for printing content deltas
        live) and, once the stream ends, appends the fully-accumulated
        assistant message — including any tool_calls — to self.messages,
        exactly like send() does for the non-streaming response.

        NOTE: this assumes llama-cpp-python streams tool_call deltas in the
        OpenAI-compatible incremental shape (chunk["choices"][0]["delta"]
        with "tool_calls": [{"index", "id", "function": {"name",
        "arguments"}}, ...], arguments arriving as partial JSON strings to
        concatenate). Print a few raw chunks from your actual version first
        to confirm this shape before relying on it — it has changed across
        llama-cpp-python releases.
        """
        self.messages.append({"role": "user", "content": content})
        reset = not self._started
        accumulated_content: list[str] = []
        tool_buffers: dict[int, dict[str, Any]] = {}

        for chunk in self.runtime.stream_chat(
            self.messages,
            reset=reset,
            config=config,
            tools=tools,
            tool_choice=tool_choice,
            response_format=response_format,
            **kwargs,
        ):
            self._started = True
            yield chunk
            delta = chunk.get("choices", [{}])[0].get("delta", {})

            piece = delta.get("content")
            if piece:
                accumulated_content.append(piece)

            for tc in delta.get("tool_calls") or []:
                idx = tc.get("index", 0)
                buf = tool_buffers.setdefault(
                    idx,
                    {
                        "id": None,
                        "type": "function",
                        "function": {"name": "", "arguments": ""},
                    },
                )
                if tc.get("id"):
                    buf["id"] = tc["id"]
                fn = tc.get("function") or {}
                if fn.get("name"):
                    buf["function"]["name"] += fn["name"]
                args = fn.get("arguments")
                if isinstance(args, str):
                    buf["function"]["arguments"] += args
                elif args:  # some versions send the whole dict at once
                    buf["function"]["arguments"] = args

        message: dict[str, Any] = {
            "role": "assistant",
            "content": "".join(accumulated_content) or None,
        }
        if tool_buffers:
            message["tool_calls"] = [tool_buffers[i] for i in sorted(tool_buffers)]
        self.messages.append(message)


class RuntimePool:
    """
    Manage a collection of named LlamaRuntime instances.

    Features
    --------
    - Direct dispatch: pool.chat("name", messages) / pool.complete("name", prompt)
    - LRU eviction: set max_loaded to cap how many models sit in RAM at once.
      When a new model needs to load and the cap is reached, the least-recently-
      used model is automatically unloaded (not destroyed — it reloads on demand).

    Example::

        pool = RuntimePool(max_loaded=2)

        pool.add("planner", "planner.gguf", RuntimeOptions.cpu_only(n_ctx=512))
        pool.add("coder",   "coder.gguf",   RuntimeOptions.auto(n_ctx=4096))
        pool.add("reviewer","reviewer.gguf",RuntimeOptions.fast())

        pool["planner"].set_system_prompt(PLAN_PROMPT)

        result = pool.chat("planner", messages)
        result = pool.chat("coder",   messages)
        # "planner" and "coder" are loaded; adding "reviewer" would evict the LRU.
    """

    def __init__(self, max_loaded: int | None = None) -> None:
        self._runtimes: dict[str, LlamaRuntime] = {}
        self._last_used: dict[str, float] = {}
        self._lock = threading.Lock()
        self.max_loaded = max_loaded

    # ── Registration ──────────────────────────────────────────────────────────

    def add(
        self,
        name: str,
        source: str | ModelSource,
        options: RuntimeOptions | None = None,
        *,
        replace: bool = False,
    ) -> LlamaRuntime:
        """
        Register a model under ``name``.

        ``source`` can be a local file path string or a ModelSource object.
        Raises KeyError if ``name`` already exists unless ``replace=True``.
        """
        if isinstance(source, str):
            source = ModelSource(model_path=source)

        with self._lock:
            if name in self._runtimes and not replace:
                raise KeyError(
                    f"Runtime {name!r} already registered. Pass replace=True to overwrite."
                )
            if name in self._runtimes:
                self._runtimes[name].close()

            rt = LlamaRuntime(source, options or RuntimeOptions.auto())
            self._runtimes[name] = rt
            self._last_used[name] = 0.0
            log.info("RuntimePool: registered %r (%s).", name, source.display_name)
            return rt

    def remove(self, name: str) -> None:
        """Close and remove a runtime from the pool."""
        with self._lock:
            rt = self._runtimes.pop(name, None)
            self._last_used.pop(name, None)
        if rt is not None:
            rt.close()
            log.info("RuntimePool: removed %r.", name)

    # ── Retrieval ─────────────────────────────────────────────────────────────

    def get(self, name: str) -> LlamaRuntime:
        try:
            return self._runtimes[name]
        except KeyError:
            raise KeyError(f"No runtime {name!r} in pool.") from None

    def __getitem__(self, name: str) -> LlamaRuntime:
        return self.get(name)

    def names(self) -> list[str]:
        with self._lock:
            return list(self._runtimes)

    def loaded_names(self) -> list[str]:
        with self._lock:
            return [n for n, rt in self._runtimes.items() if rt.loaded]

    def chat(
        self,
        name: str,
        messages: Sequence[dict[str, str]],
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Run chat() on the named runtime, handling LRU eviction if needed."""
        self._ensure_capacity(name)
        result = self.get(name).chat(messages, **kwargs)
        self._mark_used(name)
        return result

    def complete(self, name: str, prompt: str, **kwargs: Any) -> dict[str, Any]:
        """Run complete() on the named runtime, handling LRU eviction if needed."""
        self._ensure_capacity(name)
        result = self.get(name).complete(prompt, **kwargs)
        self._mark_used(name)
        return result

    async def achat(
        self,
        name: str,
        messages: Sequence[dict[str, str]],
        **kwargs: Any,
    ) -> dict[str, Any]:
        self._ensure_capacity(name)
        result = await self.get(name).achat(messages, **kwargs)
        self._mark_used(name)
        return result

    async def acomplete(self, name: str, prompt: str, **kwargs: Any) -> dict[str, Any]:
        self._ensure_capacity(name)
        result = await self.get(name).acomplete(prompt, **kwargs)
        self._mark_used(name)
        return result

    def unload_all(self) -> None:
        """Unload all models (runtimes remain registered and can reload)."""
        with self._lock:
            for rt in self._runtimes.values():
                rt.unload()

    def close_all(self) -> None:
        """Shut down every runtime and clear the pool."""
        with self._lock:
            for name, rt in self._runtimes.items():
                log.info("RuntimePool: closing %r.", name)
                rt.close()
            self._runtimes.clear()
            self._last_used.clear()

    def snapshot(self) -> dict[str, Any]:
        """Per-runtime load status plus process-level RSS if psutil is present."""
        info: dict[str, Any] = {
            n: {
                "loaded": rt.loaded,
                "last_usage": rt.last_usage,
                "last_used": self._last_used.get(n, 0.0),
            }
            for n, rt in self._runtimes.items()
        }
        if psutil is not None:
            proc = psutil.Process()
            info["_process"] = {"rss_mb": round(proc.memory_info().rss / (1024**2), 1)}
        return info

    def __enter__(self) -> "RuntimePool":
        return self

    def __exit__(self, *_: Any) -> None:
        self.close_all()

    def _ensure_capacity(self, name: str) -> None:
        """
        If max_loaded is set and loading ``name`` would exceed the cap,
        unload the least-recently-used model first.
        """
        if self.max_loaded is None:
            return
        rt = self._runtimes.get(name)
        if rt is None or rt.loaded:
            return  # already in memory — nothing to evict

        with self._lock:
            loaded = [
                (n, self._last_used.get(n, 0.0))
                for n, r in self._runtimes.items()
                if r.loaded and n != name
            ]
            if len(loaded) < self.max_loaded:
                return
            lru_name = min(loaded, key=lambda x: x[1])[0]

        log.info("RuntimePool: LRU evict %r to make room for %r.", lru_name, name)
        self._runtimes[lru_name].unload()

    def _mark_used(self, name: str) -> None:
        with self._lock:
            self._last_used[name] = time.monotonic()
