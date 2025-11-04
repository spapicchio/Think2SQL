import argparse
from huggingface_hub import try_to_load_from_cache


def get_path_from_cache(cache_dir, model_id) -> str:
    """the number of generations must be a dividend of the number of GPUs times the batch size"""
    filepath = try_to_load_from_cache(
        repo_id=model_id,
        filename="config.json",
        cache_dir=cache_dir,
    )
    if isinstance(filepath, str):
        from pathlib import Path
        # file exists and is cached
        return Path(filepath).parent

    return model_id


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--cache_dir", type=str, required=True, help="The directory of the cache"
    )
    parser.add_argument("--model_id", type=str, required=True, help="The model ID")

    args = parser.parse_args()
    model_path = get_path_from_cache(args.cache_dir, args.model_id)
    print(model_path)
