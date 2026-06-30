# main.py
import typer
from rich.traceback import install
from src.ui.commands.run import run
from src.ui.theme import console, arc_rule, arc_panel, arc_status
from src.ui.commands.doctor import doctor
from src.ui.commands.models import app as models_app

install(show_locals=False)

app = typer.Typer(
    name="arc",
    add_completion=False,
    rich_markup_mode="rich",
    help="ARC — Action & Reasoning Core",
    no_args_is_help=True,
)


_ = app.command()(run)


_ = app.command()(doctor)

app.add_typer(
    models_app,
    name="models",
)

if __name__ == "__main__":
    app()
