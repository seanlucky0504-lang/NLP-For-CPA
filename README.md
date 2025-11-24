# NLP-For-CPA: Teacher-Student Agentic Dataset Builder

This project sketches a lightweight multi-agent pipeline for generating CPA exam training data with a teacher agent (DeepSeek via LangChain) and a distilled student model (e.g., Qwen1.5-1.8B with LoRA). It also includes evaluation and a minimal Streamlit demo for side-by-side answers.

## Project layout

- `src/teacher/` – Planner, Writer, Reviewer agents and orchestration helpers for dataset synthesis.
- `src/student/` – Student model fine-tuning utilities and inference helpers.
- `src/eval/` – Automatic metrics and experiment scaffolding.
- `app.py` – Streamlit UI for comparing Teacher vs Student answers.
- `data/sample_teacher_dataset.json` – Small CPA-style QA sample for quick testing.

## Quickstart

1. Install dependencies (Python 3.10+ recommended):

   ```bash
   pip install -r requirements.txt
   ```

2. Set environment variables for the teacher model API (DeepSeek-style):

   ```bash
   export DEEPSEEK_API_BASE="https://api.deepseek.com"
   export DEEPSEEK_API_KEY="sk-..."
   ```

3. Run a small synthetic generation job to produce a dataset under `data/`:

   ```bash
   python -m src.teacher.pipeline --topic "财务成本管理" --num-questions 10
   ```

   The CLI now prints per-sample progress with ETA hints so you can estimate how long larger runs (e.g., 200–500 samples) will take. With online DeepSeek access, expect roughly 1–3 samples per second depending on latency; offline fallback runs are near-instant but produce placeholder text.

4. Fine-tune the student model with LoRA (example uses Qwen1.5-1.8B-Chat):

   ```bash
   python scripts/train_student.py --data data/sample_teacher_dataset.json --model Qwen1.5-1.8B-Chat
   ```

5. Launch the demo UI to compare Teacher and Student answers:

   ```bash
   streamlit run app.py
   ```

## Notes

- The teacher side relies on LangChain; replace the DeepSeek chat wrapper with any compatible OpenAI-style backend.
- The sample dataset is tiny and only for smoke tests; for real experiments, enlarge via the pipeline and add held-out evaluation splits.
- The student fine-tuning script now applies causal-LM padding via `DataCollatorForLanguageModeling` so variable-length samples batch correctly.
- Evaluation scaffolding provides BLEU/BERTScore hooks and human-rating placeholders; plug in your metrics of choice.
