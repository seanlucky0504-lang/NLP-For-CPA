import argparse
from pathlib import Path

from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline


def load_student(base_model: str, lora_path: Path):
    tokenizer = AutoTokenizer.from_pretrained(base_model)
    base = AutoModelForCausalLM.from_pretrained(base_model)
    model = PeftModel.from_pretrained(base, lora_path)
    return tokenizer, model


def chat(tokenizer, model, question: str, max_new_tokens: int = 256) -> str:
    generator = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
    )
    prompt = f"用户问题：{question}\n请像 CPA 老师一样回答："
    outputs = generator(prompt, max_new_tokens=max_new_tokens, do_sample=True, temperature=0.7)
    return outputs[0]["generated_text"][len(prompt) :].strip()


def main() -> None:
    parser = argparse.ArgumentParser(description="Run inference with LoRA student model")
    parser.add_argument("--model", default="Qwen1.5-1.8B-Chat")
    parser.add_argument("--lora", type=Path, default=Path("outputs/student_lora"))
    parser.add_argument("--question", required=True)
    args = parser.parse_args()

    tokenizer, model = load_student(args.model, args.lora)
    answer = chat(tokenizer, model, args.question)
    print(answer)


if __name__ == "__main__":
    main()
