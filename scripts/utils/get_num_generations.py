import argparse
from math import gcd


def get_num_generations(num_gpus: int, bs: int, max_generations: int) -> int:
    """the number of generations must be a dividend of the number of GPUs times the batch size"""
    return (
        max(gcd(num_gpus * bs, max_generations), 1)
        if num_gpus * bs % max_generations != 0
        else max_generations
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--num_gpus", type=int, required=True, help="The number of GPUs available"
    )
    parser.add_argument("--bs", type=int, required=True, help="The batch size per GPU")
    parser.add_argument(
        "--max_generations",
        type=int,
        required=True,
        help="The maximum number of generations to be used",
    )

    args = parser.parse_args()
    num_generations = get_num_generations(args.num_gpus, args.bs, args.max_generations)
    print(num_generations)
