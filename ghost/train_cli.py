"""Command line entrypoint for model training and utility evaluation."""

from __future__ import annotations

from argparse import ArgumentParser

from .train import TrainConfig, run_training


def build_parser() -> ArgumentParser:
    parser = ArgumentParser("Train and evaluate models on original or GHOST-obfuscated data")
    parser.add_argument("--data", default="sst2")
    parser.add_argument("--model_name", default="bert-base-uncased")
    parser.add_argument("--task", default="auto", choices=["auto", "classification", "generation"])
    parser.add_argument("--train_source", default="transformed", choices=["original", "transformed"])
    parser.add_argument("--transformed_data_dir", default="data")
    parser.add_argument("--results_dir", default="results")
    parser.add_argument("--models_dir", default="models")
    parser.add_argument("--topk", type=int, default=70)
    parser.add_argument("--beam_width", type=int, default=1)
    parser.add_argument("--overlap", type=float, default=0.1)
    parser.add_argument("--num_of_samples", type=int, default=1000)
    parser.add_argument("--train_ratio", type=float, default=0.8)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--learning_rate", type=float, default=1e-5)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--max_epochs", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--max_length", type=int, default=None)
    parser.add_argument("--hf_token", default=None)
    parser.add_argument("--torch_dtype", default="auto")
    parser.add_argument("--load_in_4bit", action="store_true")
    parser.add_argument("--lora", default="auto", choices=["auto", "on", "off"])
    parser.add_argument("--lora_rank", type=int, default=8)
    parser.add_argument("--lora_alpha", type=int, default=16)
    parser.add_argument("--lora_dropout", type=float, default=0.1)
    parser.add_argument("--gradient_noise", type=float, default=None)
    parser.add_argument("--gradient_prune", type=float, default=None)
    parser.add_argument("--no_save_model", action="store_true")
    parser.add_argument("--append_results", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    config = TrainConfig(
        dataset=args.data,
        model_name=args.model_name,
        task=args.task,
        train_source=args.train_source,
        transformed_data_dir=args.transformed_data_dir,
        results_dir=args.results_dir,
        models_dir=args.models_dir,
        top_k=args.topk,
        beam_width=args.beam_width,
        overlap_threshold=args.overlap,
        num_samples=args.num_of_samples,
        train_ratio=args.train_ratio,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        patience=args.patience,
        max_epochs=args.max_epochs,
        seed=args.seed,
        device=args.device,
        max_length=args.max_length,
        hf_token=args.hf_token,
        torch_dtype=args.torch_dtype,
        load_in_4bit=args.load_in_4bit,
        lora=args.lora,
        lora_rank=args.lora_rank,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        gradient_noise=args.gradient_noise,
        gradient_prune=args.gradient_prune,
        save_model=not args.no_save_model,
        overwrite_results=not args.append_results,
    )
    result_path = run_training(config)
    print(f"Wrote metrics to {result_path}")


if __name__ == "__main__":
    main()
