import os
import argparse
from huggingface_hub import try_to_load_from_cache

def get_path_from_cache(hub_cache_dir, model_id) -> str:
    # 1. Check if model_id is already a local path
    if os.path.isdir(model_id):
        return model_id
        
    filepath = try_to_load_from_cache(
        repo_id=model_id,
        filename="config.json",
        cache_dir=hub_cache_dir,
    )
    if isinstance(filepath, str):
        from pathlib import Path
        return Path(filepath).parent

    return model_id


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--hub_cache_dir", type=str, default=f"{os.getenv('HF_HOME')}/hub", help="The directory of the cache"
    )
    parser.add_argument("--model_id", type=str, required=True, help="The model ID")

    args = parser.parse_args()
    model_path = get_path_from_cache(args.hub_cache_dir, args.model_id)
    print(model_path)
