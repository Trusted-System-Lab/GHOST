"""Command line entrypoint for the reconstructed transformation pipeline."""

from __future__ import annotations

from argparse import ArgumentParser

from .pipeline import TransformConfig, run_transform
from .search import ShadowSearchConfig
from .select import SelectionConfig


def build_parser() -> ArgumentParser:
    parser = ArgumentParser("Transform data using reconstructed GHOST token obfuscation")
    parser.add_argument(
        "--data",
        default="sst2",
        help=(
            "Dataset. Classification: cola, sst2, rotten_tomatoes, tweeter, yahoo. "
            "Generation: enron, medical, legal, news, fine_persona, medical_chatbot, "
            "medical_qna, legal_task, news_dataset."
        ),
    )
    parser.add_argument("--task", default="auto", choices=["auto", "classification", "generation"])
    parser.add_argument("--model_name", default="bert-base-uncased", help="Hugging Face backbone model")
    parser.add_argument("--num_of_samples", type=int, default=1000, help="Number of class-balanced samples")
    parser.add_argument("--device", default="auto", help="Device, e.g. auto, cpu, cuda:0, mps")
    parser.add_argument("--output_dir", default="data", help="Directory for transformed data and shadow cache")
    parser.add_argument("--source_data_dir", default="../data", help="Directory containing local generation JSON data")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max_length", type=int, default=None)
    parser.add_argument("--max_words", type=int, default=None, help="Optional word-count filter for generation data")
    parser.add_argument("--recover_batch", type=int, default=0, help="Resume after this many transformed samples")
    parser.add_argument("--hf_token", default=None, help="HF token for gated models; defaults to HF_TOKEN env var")
    parser.add_argument("--add_eos_token", action="store_true", help="Set tokenizer.add_eos_token when supported")
    parser.add_argument("--load_in_4bit", action="store_true", help="Load large decoder models with bitsandbytes 4-bit")
    parser.add_argument("--torch_dtype", default="auto", help="auto, float32, float16, or bfloat16")

    parser.add_argument("--topk", type=int, default=70, help="Initial embedding-neighbor search size")
    parser.add_argument("--overlap", type=float, default=0.1, help="Indirect-similarity overlap threshold")
    parser.add_argument("--max_search_k", type=int, default=None, help="Maximum expanded neighbor search size")
    parser.add_argument("--similarity_batch_size", type=int, default=512)
    parser.add_argument("--overwrite_shadow_cache", action="store_true")

    parser.add_argument("--beam_width", type=int, default=1)
    parser.add_argument("--early_stop_delta", type=float, default=0.1)
    parser.add_argument("--max_iterations", type=int, default=20)
    parser.add_argument("--eval_batch_size", type=int, default=64)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    search = ShadowSearchConfig(
        top_k=args.topk,
        overlap_threshold=args.overlap,
        max_search_k=args.max_search_k,
        similarity_batch_size=args.similarity_batch_size,
    )
    selection = SelectionConfig(
        beam_width=args.beam_width,
        early_stop_delta=args.early_stop_delta,
        max_iterations=args.max_iterations,
        eval_batch_size=args.eval_batch_size,
    )
    config = TransformConfig(
        dataset=args.data,
        model_name=args.model_name,
        task=args.task,
        num_samples=args.num_of_samples,
        output_dir=args.output_dir,
        source_data_dir=args.source_data_dir,
        device=args.device,
        seed=args.seed,
        max_length=args.max_length,
        max_words=args.max_words,
        recover_batch=args.recover_batch,
        hf_token=args.hf_token,
        add_eos_token=args.add_eos_token,
        load_in_4bit=args.load_in_4bit,
        torch_dtype=args.torch_dtype,
        overwrite_shadow_cache=args.overwrite_shadow_cache,
        search=search,
        selection=selection,
    )
    output_path = run_transform(config)
    print(f"Wrote transformed data to {output_path}")


if __name__ == "__main__":
    main()
