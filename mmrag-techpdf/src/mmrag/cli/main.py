import typer

app = typer.Typer()


@app.command()
def index(pdf_path: str, output_dir: str = "data/indexes"):
    """Index a PDF document."""
    typer.echo(f"Indexing {pdf_path} to {output_dir}")


@app.command()
def query(question: str, index_dir: str = "data/indexes"):
    """Query the indexed documents."""
    typer.echo(f"Querying: {question} using index at {index_dir}")


if __name__ == "__main__":
    app()