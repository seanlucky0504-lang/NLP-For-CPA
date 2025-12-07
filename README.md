# NLP-For-CPA: Teacher-Student Agentic Dataset Builder

This project sketches a lightweight multi-agent pipeline for generating CPA exam training data with a teacher agent (DeepSeek via LangChain) and a distilled student model (e.g., Qwen1.5-1.8B with LoRA). It also includes evaluation and a minimal Streamlit demo for side-by-side answers.

## Project layout

- `src/teacher/` – Planner, Writer, Reviewer agents and orchestration helpers for dataset synthesis.
- `src/student/` – Student model fine-tuning utilities and inference helpers.
- `src/eval/` – Automatic metrics and experiment scaffolding.
- `scripts/` – CLI helpers for dataset synthesis, fine-tuning, and evaluation.
- `app.py` – Streamlit UI for comparing Teacher vs Student answers.
- `data/sample_teacher_dataset.json` – Small CPA-style QA sample for quick testing.

## Quickstart: run end-to-end

1. Install dependencies (Python 3.10+ recommended):

   ```bash
   pip install -r requirements.txt
   ```

2. Set environment variables for the teacher model API (DeepSeek-style):

   ```bash
   export DEEPSEEK_API_BASE="https://api.deepseek.com"
   export DEEPSEEK_API_KEY="sk-xxxxxxxx"
   ```

3. 合成并“蒸馏”高质量教师数据（评分过滤，默认阈值 7 分）到 JSONL：

   ```bash
   python scripts/generate_dataset.py --topic "财务成本管理" --num-questions 500 --min-score 7.5 --jsonl
   # 输出示例：data/generated/财务成本管理_teacher_500.jsonl
   ```

4. 使用 LoRA/QLoRA 微调小模型（自动切分验证集）：

   ```bash
   python scripts/train_student.py \
     --data data/generated/财务成本管理_teacher_500.jsonl \
     --model Qwen1.5-1.8B-Chat \
     --output-dir outputs/student_lora \
     --qlora \
     --epochs 3 --batch-size 2 --gradient-accumulation-steps 4
   ```

5. 生成学生答案并对比教师参考，输出指标：

   ```bash
   python scripts/eval_student.py \
     --teacher data/generated/财务成本管理_teacher_500.jsonl \
     --model Qwen1.5-1.8B-Chat \
     --lora outputs/student_lora \
     --limit 200
   ```

6. 打开可视化对比界面（无需改动当前架构）：

   ```bash
   streamlit run app.py
   ```

## Notes

- Teacher 侧使用 LangChain + DeepSeek API，可替换为任意兼容 OpenAI 的后端。
- `scripts/generate_dataset.py` 会自动丢弃评分低于阈值的 QA，确保蒸馏后训练集质量。
- 训练脚本支持 JSON/JSONL 数据、验证切分和 QLoRA，低显存也可跑通。
- `scripts/eval_student.py` 会生成学生预测、对齐教师参考并计算 BLEU/BERTScore，便于快速评估。
