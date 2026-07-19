import re
from math import ceil

# --- explicit structured content ---
_CODE_BLOCK_RE = re.compile(r"```[\s\S]*?```", re.MULTILINE)
_INLINE_CODE_RE = re.compile(r"`[^`\n]{3,}`")

# --- urls ---
_URL_RE = re.compile(
    r"(?<!\w)(?:https?://|www\.)[^\s<>'\"()\[\]]+[^\s<>'\"()\[\].,;:!?]",
    re.I,
)

# --- paths ---
_WINDOWS_PATH_RE = re.compile(
    r"(?<![\w/])"
    r"(?:[A-Za-z]:\\(?:[^\\/:*?\"<>|\r\n]+\\)*[^\\/:*?\"<>|\r\n]+)"
)

_UNIX_PATH_RE = re.compile(
    r"(?<![\w/])"
    r"(?:~|\.{1,2}|/)"
    r"(?:/|[\w.\-])+"
    r"(?:\.\w+)?"
)

_WHITESPACE_RE = re.compile(r"\s+")

# --- code-like lines / chunks ---
_TRACEBACK_START_RE = re.compile(r"^\s*Traceback \(most recent call last\):\s*$")
_FILE_LINE_RE = re.compile(r'^\s*File\s+"[^"]+", line \d+')
_INDENTED_LINE_RE = re.compile(r"^\s{4,}\S")
_PY_IMPORT_RE = re.compile(r"^\s*(from\s+\S+\s+import\s+|import\s+\S+)\b")
_PY_DEF_RE = re.compile(r"^\s*(def|class)\s+\w+\b")
_PY_FLOW_RE = re.compile(r"^\s*(if|for|while|try|except|with|elif|else)\b")
_PY_RETURN_RE = re.compile(r"^\s*return\b")
_SHELL_PROMPT_RE = re.compile(r"^\s*(?:\$|>|❯)\s+")
_SHELL_CMD_RE = re.compile(
    r"^\s*(?:"
    r"uv|python|python3|pip|pip3|pipx|git|docker|podman|make|pytest|"
    r"npm|yarn|pnpm|cargo|go|bash|sh|zsh|fish|curl|wget|ls|cd|cat|grep|sed|awk"
    r")\b"
)

# lines that often appear in pasted code / stack traces
_CODE_PUNCT_RE = re.compile(r"[{}[\]();=<>]|::|->|=>|:=|\\n")


def normalize_text(text: str) -> str:
    return _WHITESPACE_RE.sub(" ", text).strip()


def _mask_spans(
    text: str, pattern: re.Pattern[str], token: str
) -> tuple[str, list[str]]:
    parts: list[str] = []
    out: list[str] = []
    last = 0
    for i, m in enumerate(pattern.finditer(text)):
        parts.append(m.group(0))
        out.append(text[last : m.start()])
        out.append(f" {token}{i} ")
        last = m.end()
    out.append(text[last:])
    return "".join(out), parts


def _chunk_text(text: str) -> list[str]:
    chunks: list[str] = []
    current: list[str] = []

    for line in text.splitlines():
        if line.strip():
            current.append(line)
        else:
            if current:
                chunks.append("\n".join(current))
                current = []

    if current:
        chunks.append("\n".join(current))

    return chunks


def _looks_like_code_line(line: str) -> bool:
    s = line.rstrip()
    if not s:
        return False

    if _TRACEBACK_START_RE.match(s):
        return True
    if _FILE_LINE_RE.match(s):
        return True
    if _SHELL_PROMPT_RE.match(s):
        return True
    if _PY_IMPORT_RE.match(s):
        return True
    if _PY_DEF_RE.match(s):
        return True
    if _PY_FLOW_RE.match(s):
        return True
    if _PY_RETURN_RE.match(s):
        return True
    if _SHELL_CMD_RE.match(s):
        return True

    if _INDENTED_LINE_RE.match(s) and _CODE_PUNCT_RE.search(s):
        return True

    if _CODE_PUNCT_RE.search(s) and len(s) <= 120:
        # code-ish punctuation, but avoid ordinary prose like "I ran:"
        if any(
            ch in s for ch in ("=", "(", ")", "{", "}", "[", "]", ";", "->", "::", ":=")
        ):
            return True

    return False


def _is_code_chunk(chunk: str) -> bool:
    lines = [line for line in chunk.splitlines() if line.strip()]
    if not lines:
        return False

    if any(
        _TRACEBACK_START_RE.match(line) or _FILE_LINE_RE.match(line) for line in lines
    ):
        return True

    score = 0
    for line in lines:
        if _looks_like_code_line(line):
            score += 1

    # Multi-line chunks need multiple signals to avoid false positives.
    if len(lines) >= 2 and score >= max(2, ceil(len(lines) * 0.5)):
        return True

    # Single-line code/command snippets
    if len(lines) == 1 and _looks_like_code_line(lines[0]):
        return True

    return False


def strip_context_noise(text: str) -> str:
    # Remove explicit fenced/inline code and URLs first.
    text = _CODE_BLOCK_RE.sub(" ", text)
    text = _INLINE_CODE_RE.sub(" ", text)
    text = _URL_RE.sub(" ", text)

    # Mask URLs before path detection so URL slashes never become paths.
    masked = text

    # Replace URLs with placeholders and remember them only for stripping.
    url_matches = list(_URL_RE.finditer(masked))
    for i, m in enumerate(reversed(url_matches)):
        masked = masked[: m.start()] + f" __URL{i}__ " + masked[m.end() :]

    # Remove paths on the masked text.
    masked = _WINDOWS_PATH_RE.sub(" ", masked)
    masked = _UNIX_PATH_RE.sub(" ", masked)

    # Remove code-like pasted chunks without fences.
    kept_chunks: list[str] = []
    for chunk in _chunk_text(masked):
        if _is_code_chunk(chunk):
            continue
        kept_chunks.append(chunk)

    return normalize_text(" ".join(kept_chunks))


def extract_context_items(text: str) -> list[str]:
    items: list[str] = []
    seen: set[str] = set()

    def add(tag: str, value: str) -> None:
        value = normalize_text(value)
        if not value:
            return
        key = f"{tag}:{value.lower()}"
        if key in seen:
            return
        seen.add(key)
        items.append(f"{tag}: {value}")

    # Explicit fenced / inline code first
    for m in _CODE_BLOCK_RE.finditer(text):
        add("code", m.group().strip("`"))
    for m in _INLINE_CODE_RE.finditer(text):
        add("code", m.group().strip("`"))

    # URLs first, so paths can't eat URL text
    masked = text
    url_spans = list(_URL_RE.finditer(text))
    for i, m in enumerate(reversed(url_spans)):
        add("url", m.group())
        masked = masked[: m.start()] + f" __URL{i}__ " + masked[m.end() :]

    # Paths on text with URLs masked out
    for m in _WINDOWS_PATH_RE.finditer(masked):
        add("path", m.group())
    for m in _UNIX_PATH_RE.finditer(masked):
        add("path", m.group())

    # Heuristic detection for pasted code / shell / traceback blocks
    for chunk in _chunk_text(text):
        if _is_code_chunk(chunk):
            add("code", chunk)

    return items


def test_case(name: str, text: str):
    print("=" * 80)
    print(name)
    print("- Input:")
    print(text)
    print("\n- Context items:")
    for item in extract_context_items(text):
        print(f"  {item}")
    print("\n- Stripped:")
    print(strip_context_noise(text))
    print()


if __name__ == "__main__":
    test_case(
        "Plain text with pasted URL",
        "Can you review https://github.com/ggml-org/llama.cpp for me?",
    )

    test_case(
        "Pasted Python without fences",
        """I'm getting this: from pathlib import Path
print(Path.cwd())

Can you help?""",
    )

    test_case(
        "Shell session",
        """I ran:

uv sync
python main.py
git status

and got an error.""",
    )

    test_case(
        "Traceback",
        """Traceback (most recent call last):
  File "/home/paul/test.py", line 5, in <module>
    main()
RuntimeError: boom""",
    )

    test_case(
        "Mixed paste",
        """Repo: https://github.com/user/project
File: /home/paul/project/main.py

from pathlib import Path
print(Path.home())

Thanks.""",
    )
