# main.py
import typer
from rich.traceback import install
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


@app.command()
def run():
    console.print(arc_rule("ARC"))
    console.print(arc_panel("Runtime ready", title="Status", border_style="arc.violet"))
    console.print(arc_status("ok", "Loaded local model"))


app.command()(doctor)

app.add_typer(
    models_app,
    name="models",
)

if __name__ == "__main__":
    app()
