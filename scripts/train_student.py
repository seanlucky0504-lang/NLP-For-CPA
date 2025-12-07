import argparse
import inspect
from pathlib import Path
from typing import Dict

from datasets import load_dataset
from peft import LoraConfig, get_peft_model
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    DataCollatorForLanguageModeling,
    Trainer,
    TrainingArguments,
)


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="LoRA/QLoRA fine-tune a student model on teacher QA data")
    parser.add_argument("--data", type=Path, required=True, help="Path to teacher JSON/JSONL dataset")
    parser.add_argument("--model", type=str, default="Qwen1.5-1.8B-Chat", help="Base model name")
    parser.add_argument("--output-dir", type=Path, default=Path("./outputs/student_lora"))
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--val-ratio", type=float, default=0.05, help="Validation split ratio")
    parser.add_argument("--eval-size", type=int, default=128, help="Max validation examples (speed-up)")
    parser.add_argument("--gradient-accumulation-steps", type=int, default=4)
    parser.add_argument("--qlora", action="store_true", help="Enable 4-bit QLoRA (requires bitsandbytes)")
    return parser


def preprocess_function(tokenizer, max_length: int):
    def _inner(example: Dict[str, str]):
        prompt = f"用户问题：{example['input']}\n请像 CPA 老师一样回答："
        text = prompt + example["output"]
        tokens = tokenizer(
            text,
            truncation=True,
            max_length=max_length,
            padding="max_length",  # 保证批次对齐，避免 DataCollator 抛 padding 错误
        )
        tokens["labels"] = tokens["input_ids"].copy()
        return tokens

    return _inner


def train(args: argparse.Namespace) -> None:
    dataset = load_dataset("json", data_files=str(args.data))
    if args.val_ratio > 0:
        split = dataset["train"].train_test_split(test_size=args.val_ratio, seed=42)
        dataset = {"train": split["train"]}
        dataset["validation"] = split["test"].select(range(min(args.eval_size, len(split["test"]))))
    else:
        dataset = {"train": dataset["train"]}

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    quant_config = (
        BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype="bfloat16",
        )
        if args.qlora
        else None
    )

    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        device_map="auto",
        quantization_config=quant_config,
    )

    peft_config = LoraConfig(
        r=8,
        lora_alpha=16,
        lora_dropout=0.1,
        target_modules=["q_proj", "v_proj"],
    )
    model = get_peft_model(model, peft_config)

    tokenized_train = dataset["train"].map(
        preprocess_function(tokenizer, args.max_length),
        remove_columns=dataset["train"].column_names,
    )
    tokenized_eval = (
        dataset.get("validation").map(
            preprocess_function(tokenizer, args.max_length),
            remove_columns=dataset["train"].column_names,
        )
        if "validation" in dataset
        else None
    )

    ta_kwargs = dict(
        output_dir=str(args.output_dir),
        per_device_train_batch_size=args.batch_size,
        num_train_epochs=args.epochs,
        learning_rate=args.learning_rate,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        logging_steps=10,
        save_steps=200,
        evaluation_strategy="steps" if tokenized_eval else "no",
        eval_steps=200,
    )
    # 兼容老版本 transformers，过滤不支持的参数
    valid_keys = set(inspect.signature(TrainingArguments).parameters.keys())
    ta_kwargs = {k: v for k, v in ta_kwargs.items() if k in valid_keys}
    training_args = TrainingArguments(**ta_kwargs)

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_train,
        eval_dataset=tokenized_eval,
        data_collator=DataCollatorForLanguageModeling(tokenizer, mlm=False),
    )
    trainer.train()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    trainer.save_model(str(args.output_dir))
    tokenizer.save_pretrained(str(args.output_dir))


if __name__ == "__main__":
    parser = build_argparser()
    train(parser.parse_args())
