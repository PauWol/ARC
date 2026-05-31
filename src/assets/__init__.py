from pathlib import Path

from llama_cpp import LlamaGrammar

_json_gram_singleton: None | LlamaGrammar = None


def json_grammar() -> LlamaGrammar:
    """Return the LlamaGrammar for Json."""
    global _json_gram_singleton
    if _json_gram_singleton:
        return _json_gram_singleton

    g_p = Path(__file__).with_name("json.gbnf")
    j_g_i = LlamaGrammar.from_string(g_p.read_text(encoding="utf-8"))

    _json_gram_singleton = j_g_i

    return _json_gram_singleton
