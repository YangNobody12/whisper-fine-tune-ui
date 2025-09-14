import argparse
import os
from dataset import load_and_prepare_dataset
from train import train_model
import sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")




def cli_train(
    project,
    model="openai/whisper-small",
    language="english",
    output_dir="./whisper-small-finetuned",
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    max_steps=5000,
    save_steps=1000,
    eval_steps=1000,
    lr=1e-5,
):
    proj_path = os.path.join("dataset", project)
    json_path = os.path.join(proj_path, "metadata.json")
    csv_path = os.path.join(proj_path, "metadata.csv")

    if not os.path.exists(json_path) and os.path.exists(csv_path):
        print("⚙️ Converting metadata.csv → metadata.json ...")
        return f"❌ No metadata.json or metadata.csv found in {project}"

    if not os.path.exists(json_path):
        return f"❌ No metadata.json or metadata.csv found in {project}"

    print("⚙️ Preparing dataset...")
    dataset, processor, tokenizer, feature_extractor = load_and_prepare_dataset(json_path)

    print("⚙️ Start training...")
    result = train_model(
        dataset,
        processor,
        tokenizer,
        model=model,
        language=language,
        per_device_train_batch_size=per_device_train_batch_size,
        per_device_eval_batch_size=per_device_eval_batch_size,
        output_dir=output_dir,
        max_steps=max_steps,
        save_steps=save_steps,
        eval_steps=eval_steps,
        lr=lr,
    )
    return result


def main():
    parser = argparse.ArgumentParser(description="Train Whisper Model CLI")

    parser.add_argument("--project", type=str, required=True)
    parser.add_argument("--model", type=str, default="openai/whisper-small")
    parser.add_argument("--language", type=str, default="english")
    parser.add_argument("--output_dir", type=str, default="./whisper-small-finetuned")
    parser.add_argument("--per_device_train_batch_size", type=int, default=16)
    parser.add_argument("--per_device_eval_batch_size", type=int, default=16)
    parser.add_argument("--max_steps", type=int, default=5000)
    parser.add_argument("--save_steps", type=int, default=1000)
    parser.add_argument("--eval_steps", type=int, default=1000)
    parser.add_argument("--learning_rate", type=float, default=1e-5)

    args = parser.parse_args()
    result = cli_train(
        args.project,
        model=args.model,
        language=args.language,
        output_dir=args.output_dir,
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=args.per_device_eval_batch_size,
        max_steps=args.max_steps,
        save_steps=args.save_steps,
        eval_steps=args.eval_steps,
        lr=args.learning_rate,
    )
    print(result)


if __name__ == "__main__":
    main()
