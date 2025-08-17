# deep_image_analyzer/cli/main.py

# Import CLI frameworks and async support
import click
import asyncio
from pathlib import Path
from analysis.prediction import InferenceService
from utils.io import async_load_image
from core.preprocessing import get_standard_transforms

@click.command()
@click.argument("images", nargs=-1, type=click.Path(exists=True))
@click.option("--model", default="resnet50", help="Model name to use")
@click.option("--output", default="./outputs", help="Directory for output files")
def main(images, model, output):
    """
    Entry point for the command-line interface.
    """
    asyncio.run(_process_all(images, model, output))

async def _process_all(images, model_name, output_dir):
    """
    Asynchronously load and process multiple images in parallel.
    """
    svc = InferenceService(model_name)              # Initialize inference service
    transform = get_standard_transforms()           # Standard preprocessing
    tasks = []
    for img_path in images:
        tasks.append(_process_one(Path(img_path), svc, transform, Path(output_dir)))
    await asyncio.gather(*tasks)                    # Run tasks concurrently

async def _process_one(path, svc, transform, out_dir):
    """
    Load a single image, run inference, and save raw results.
    """
    img = await async_load_image(path)              # Async load
    tensor = transform(img).unsqueeze(0)            # Preprocess and add batch dim
    probs = svc.predict(tensor)[0].cpu().numpy()    # Predict and get numpy array
    # Create output directory if needed
    out_dir.mkdir(parents=True, exist_ok=True)
    # (Placeholder) Save or log `probs`, e.g. to JSON or image overlays
    # e.g.: (out_dir / f"{path.stem}_probs.json").write_text(json.dumps(probs.tolist()))

# Run CLI when module is executed directly
if __name__ == "__main__":
    main()
