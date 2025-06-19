import click
from llm_judge.code_pipeline import CodePipeline
from llm_judge.text_pipeline import TextPipeline

@click.command()
@click.argument("prompt")
@click.option(
    "--model", "-m",
    multiple=True,
    required=True,
    help="One or more model identifiers (e.g., 'gpt-4', 'llama-3')."
)
@click.option(
    "--type", "-t",
    type=click.Choice(["code", "text"]),
    required=True,
    help="Whether to evaluate code or text."
)
def main(prompt, model, type):
    """
    CLI for LLM-Judge: submit a prompt, one or more models, and pick code/text evaluation.
    """
    if type == "code":
        pipeline = CodePipeline(models=list(model))
    else:
        pipeline = TextPipeline(models=list(model))

    results = pipeline.evaluate(prompt)
    click.echo(results)

if __name__ == "__main__":
    main()
