import click
import json
import os
import uuid
from datetime import datetime
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
@click.option(
    "--save", is_flag=True,
    help="Save the evaluation results to a JSON file."
)

@click.option(
    "--batch-file",
    type=click.Path(exists=True),
    help="Path to a JSON file containing a list of prompts for batch evaluation."
)

def main(prompt, model, type, save, batch_file):
    models = list(model)

    if batch_file:
        with open(batch_file, "r") as f:
            prompts = json.load(f)

        pipeline = CodePipeline(models=models) if type == "code" else TextPipeline(models=models)
        all_results = []
        wins = {}

        for idx, prompt in enumerate(prompts):
            print(f"\n🚀 Evaluating prompt {idx + 1}/{len(prompts)}:\n📌 {prompt}")
            result = pipeline.evaluate(prompt)
            result["prompt_id"] = str(uuid.uuid4())
            all_results.append(result)

            winner = result.get("winner")
            if winner:
                wins[winner] = wins.get(winner, 0) + 1

            if save:
                timestamp = datetime.now().isoformat(timespec="seconds").replace(":", "-")
                filename = f"evaluations/eval_code_{idx+1}_{timestamp}.json"
                path = os.path.join(os.path.dirname(__file__), "..", filename)
                with open(path, "w") as f:
                    json.dump(result, f, indent=2)
                click.echo(f"💾 Saved: {filename}")

        # Final Summary
        click.echo("\n🏁 Batch Evaluation Summary:\n")
        for model, count in wins.items():
            click.echo(f"🏆 {model}: {count} wins")

    else:
        pipeline = CodePipeline(models=models) if type == "code" else TextPipeline(models=models)
        result = pipeline.evaluate(prompt)

        click.echo("\n✅ Final Result:\n")
        click.echo(json.dumps(result, indent=2))

        if save:
            timestamp = datetime.now().isoformat(timespec="seconds").replace(":", "-")
            filename = f"evaluations/eval_{timestamp}.json"
            path = os.path.join(os.path.dirname(__file__), "..", filename)
            with open(path, "w") as f:
                json.dump(result, f, indent=2)
            click.echo(f"💾 Evaluation saved to: {filename}")

if __name__ == "__main__":
    main()
