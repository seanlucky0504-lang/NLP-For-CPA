import argparse
from pathlib import Path
from typing import Dict

from datasets import load_dataset
from peft import LoraConfig, get_peft_model
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    DataCollatorForLanguageModeling,
    Trainer,
    TrainingArguments,
)


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="LoRA fine-tune a student model on teacher QA data")
    parser.add_argument("--data", type=Path, required=True, help="Path to teacher JSON dataset")
    parser.add_argument("--model", type=str, default="Qwen1.5-1.8B-Chat", help="Base model name")
    parser.add_argument("--output-dir", type=Path, default=Path("./outputs/student_lora"))
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--max-length", type=int, default=512)
    return parser


def preprocess_function(tokenizer, max_length: int):
    def _inner(example: Dict[str, str]):
        prompt = f"用户问题：{example['input']}\n请像 CPA 老师一样回答："
        text = prompt + example["output"]
        tokens = tokenizer(
            text,
            truncation=True,
            max_length=max_length,
        )
        tokens["labels"] = tokens["input_ids"].copy()
        return tokens

    return _inner


def train(args: argparse.Namespace) -> None:
    dataset = load_dataset("json", data_files=str(args.data))
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    if tokenizer.pad_token is None and tokenizer.eos_token:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(args.model)
    model.config.pad_token_id = tokenizer.pad_token_id

    peft_config = LoraConfig(
        r=8,
        lora_alpha=16,
        lora_dropout=0.1,
        target_modules=["q_proj", "v_proj"],
    )
    model = get_peft_model(model, peft_config)

    tokenized_dataset = dataset.map(
        preprocess_function(tokenizer, args.max_length),
        remove_columns=dataset["train"].column_names,
    )

    training_args = TrainingArguments(
        output_dir=str(args.output_dir),
        per_device_train_batch_size=args.batch_size,
        num_train_epochs=args.epochs,
        logging_steps=10,
        save_steps=200,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset["train"],
        data_collator=DataCollatorForLanguageModeling(
            tokenizer=tokenizer,
            mlm=False,
        ),
    )
    trainer.train()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    trainer.save_model(str(args.output_dir))
    tokenizer.save_pretrained(str(args.output_dir))


if __name__ == "__main__":
    parser = build_argparser()
    train(parser.parse_args())
