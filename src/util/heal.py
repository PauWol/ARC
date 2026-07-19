from collections.abc import Iterable

from src.constants import ENV_PATH, DEFAULT_DOT_ENV


def ensure_dot_env() -> bool:
    """Test if the '.env' exstists"""
    return ENV_PATH.exists()


def find_key(text: str, keys: Iterable[str]) -> str | None:
    return next((k for k in keys if k in text), None)


def check_missing_dot_env_entrys():
    """
    Check for missing or wrong '.env' entrys.

    :returns: tuple[list,list] of first wrong lines and second the missing entrys.
    """

    _w = []
    _e = []

    _d_env_keys = DEFAULT_DOT_ENV.keys()

    with ENV_PATH.open() as f:
        for l in f:
            l = l.strip()

            _key = find_key(l, _d_env_keys)

            if not _key:  # Wrong entrys
                _w.append(l)

            else:
                _e.append(_key)  # append exsisting

    if len(_d_env_keys) == len(_e) and len(_w) == 0:  # fast path
        return _w, None

    _m_n = []

    for e in _d_env_keys:
        if e not in _e:
            _m_n.append(e)

    return _w, _m_n


def repair_dot_env(missing: list[str], wrong: list[str]):
    if not ENV_PATH.exists():
        ENV_PATH.write_text("")

    _lines = ENV_PATH.read_text().splitlines()
    _l_out = _lines.copy()

    for l in _lines:
        if l in wrong:
            _l_out.remove(l)

    for m in missing:
        default_value = DEFAULT_DOT_ENV.get(m, "")
        _l_out.append(f'{m} = "{default_value}"')

    ENV_PATH.write_text("\n".join(_l_out) + "\n")
