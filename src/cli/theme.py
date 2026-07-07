from typing import Literal
from rich.console import Console, RenderableType
from rich.theme import Theme
from rich.style import Style
from rich.panel import Panel
from rich.text import Text
from rich.rule import Rule

# Core palette from the ARC logo
ARC_BG = "#050507"
ARC_CYAN = "#00CDFE"
ARC_CYAN_SOFT = "#0CA3D9"
ARC_BLUE = "#0765D9"
ARC_BLUE_DARK = "#144FA5"
ARC_VIOLET = "#7A22E2"
ARC_VIOLET_DARK = "#5F23B1"
ARC_TEXT = "#D6EBFB"
ARC_MUTED = "#9EC0E0"
ARC_GREEN = "#00E676"
# Semantic colors
ARC_OK = ARC_GREEN
ARC_INFO = ARC_BLUE
ARC_WARN = ARC_VIOLET
ARC_ERROR = "#FF5C8A"

# Rich theme
ARC_THEME = Theme(
    {
        "arc.primary": ARC_CYAN,
        "arc.primary.soft": ARC_CYAN_SOFT,
        "arc.blue": ARC_BLUE,
        "arc.blue.dark": ARC_BLUE_DARK,
        "arc.violet": ARC_VIOLET,
        "arc.violet.dark": ARC_VIOLET_DARK,
        "arc.green": ARC_GREEN,
        "arc.text": ARC_TEXT,
        "arc.muted": ARC_MUTED,
        "arc.ok": ARC_OK,
        "arc.info": ARC_INFO,
        "arc.warn": ARC_WARN,
        "arc.error": ARC_ERROR,
    }
)

console = Console(theme=ARC_THEME, highlight=False, soft_wrap=True)

# Common styles
TITLE_STYLE = Style(color=ARC_CYAN, bold=True)
SUBTITLE_STYLE = Style(color=ARC_MUTED)
ACCENT_STYLE = Style(color=ARC_VIOLET, bold=True)
DIM_STYLE = Style(color=ARC_MUTED, dim=True)


def arc_rule(title: str = "ARC") -> Rule:
    return Rule(title, style="arc.primary")


def arc_panel(
    renderable: RenderableType,
    title: str | None = None,
    subtitle: str | None = None,
    border_style: str = "arc.blue",
):
    return Panel(
        renderable,
        title=title,
        subtitle=subtitle,
        border_style=border_style,
        padding=(1, 2),
    )


def arc_text(label: str, value: str) -> Text:
    t = Text()
    _ = t.append(label, style="arc.muted")
    _ = t.append(value, style="arc.text")
    return t


def arc_status(
    kind: Literal["ok", "info", "warn", "error", "listing"],
    message: str,
) -> Text:
    styles = {
        "ok": "arc.ok",
        "info": "arc.info",
        "listing": "arc.info",
        "warn": "arc.warn",
        "error": "arc.error",
    }

    icons = {"ok": "✓ ", "info": "ℹ ", "warn": "⚠ ", "error": "✗ ", "listing": "● "}

    t = Text()
    _ = t.append(icons.get(kind, "● "), style=styles.get(kind, "arc.info"))
    _ = t.append(message, style="arc.text")
    return t
