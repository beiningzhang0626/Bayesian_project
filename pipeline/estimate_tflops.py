#!/usr/bin/env python3
import json
import argparse

def count_words(text: str) -> int:
    return len(text.strip().split())

def estimate_tokens_and_tflops(
    json_path: str,
    params_in_billion: float,
    tokens_per_word: float = 5.0 / 4.0,
):
    with open(json_path, "r") as f:
        data = json.load(f)

    total_words = 0
    total_tokens = 0.0
    total_pairs = 0

    for item in data["individual_results"]:
        question = item["question"]
        choices = item["choices"]

        q_words = count_words(question)

        for choice in choices:
            c_words = count_words(choice)
            pair_words = q_words + c_words

            total_words += pair_words
            total_tokens += pair_words * tokens_per_word
            total_pairs += 1

    # Model params
    num_params = params_in_billion * 1e9  # convert B -> raw count

    # FLOPs ≈ 2 * params * tokens
    total_flops = 2.0 * num_params * total_tokens
    total_tflops = total_flops / 1e12

    print(f"Model size: {params_in_billion:.3f}B parameters")
    print(f"Num (question, choice) pairs: {total_pairs}")
    print(f"Total words (question+choice, counting repeated context): {total_words}")
    print(f"Estimated total tokens: {total_tokens:.2f}")
    print(f"Estimated total FLOPs: {total_flops:.3e}")
    print(f"Estimated total TFLOPs: {total_tflops:.2f}")

    # Also return values in case you want to import as a module
    return {
        "total_pairs": total_pairs,
        "total_words": total_words,
        "total_tokens": total_tokens,
        "total_flops": total_flops,
        "total_tflops": total_tflops,
    }

def main():
    parser = argparse.ArgumentParser(
        description=(
            "Estimate TFLOPs for log-prob evaluation over a JSON results file.\n"
            "Assumes each prompt is question + one choice, and question is "
            "re-evaluated for each choice."
        )
    )
    parser.add_argument("json_path", type=str, help="Path to the JSON file")
    parser.add_argument(
        "--params-b",
        type=float,
        required=True,
        help="Model size in billions of parameters (e.g. 70.6 for Llama3.1-70B)",
    )
    parser.add_argument(
        "--tokens-per-word",
        type=float,
        default=5.0 / 4.0,
        help="Approximate tokens per word (default: 1.25 = 5/4)",
    )

    args = parser.parse_args()
    estimate_tokens_and_tflops(
        json_path=args.json_path,
        params_in_billion=args.params_b,
        tokens_per_word=args.tokens_per_word,
    )

if __name__ == "__main__":
    main()