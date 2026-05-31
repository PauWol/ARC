from src.util.constants import ENV_PATH, CONST, DEFAULT_DOT_ENV


def ensure_dot_env() -> bool:
    """Test if the '.env' exstists"""
    return ENV_PATH.exists()


def _in_const(line: str) -> str | None:
    for e in CONST:
        if line.startswith(e.value):
            return e.value
    return None


def check_missing_dot_env_entrys() -> str:
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

    if len(CONST) == len(_e):  # fast path
        return _w, None

    _m_n = []

    for e in CONST:
        if e.value not in _e:
            _m_n.append(e)

    return _w, _m_n


def repair_dot_env(missing: list[str], wrong: list[str]):
    if not ENV_PATH.exists():
        ENV_PATH.write_text("")

    _lines = ENV_PATH.read_text().splitlines()

    for l in _lines:
        if l in wrong:
            del l
            wrong.remove(l)

    for m in missing:
        default_value = DEFAULT_DOT_ENV.get(CONST(m), "")
        _lines.append(f"{m}={default_value}")
        _lines.append(f"{m} = {DEFAULT_DOT_ENV.get(m)}")

    ENV_PATH.write_text("\n".join(_lines) + "\n")
