from src.util.constants import ENV_PATH, CONST, DEFAULT_DOT_ENV


def ensure_dot_env() -> bool:
    """Test if the '.env' exstists"""
    return ENV_PATH.exists()


def _in_const(line: str) -> str | None:
    for e in CONST:
        if line.startswith(e.value):
            return e.value
    return None


def check_missing_dot_env_entrys():
    """
    Check for missing or wrong '.env' entrys.

    :returns: tuple[list,list] of first wrong lines and second the missing entrys.
    """

    _w = []
    _e = []

    with ENV_PATH.open() as f:
        for l in f:
            l = l.strip()

            key = _in_const(l)

            if not key:  # Wrong entrys
                _w.append(l)

            else:
                _e.append(key)  # append exsisting

    _d_env_keys = DEFAULT_DOT_ENV.keys()

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
