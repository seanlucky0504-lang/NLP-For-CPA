import json
import os
from pathlib import Path

import streamlit as st

from src.teacher.agents import DatasetSynthesizer, PlannerAgent, ReviewerAgent, WriterAgent
from src.student.inference import chat, load_student

st.set_page_config(page_title="CPA Teacher vs Student", page_icon="📚")

st.title("CPA Teacher vs Student 对比")

with st.sidebar:
    st.header("模型配置")
    deepseek_ready = bool(os.getenv("DEEPSEEK_API_KEY") and os.getenv("DEEPSEEK_API_BASE"))
    st.write("Teacher API 已配置" if deepseek_ready else "Teacher API 未配置，使用占位输出")
    student_base = st.text_input("Student 基座模型", value="Qwen1.5-1.8B-Chat")
    lora_path = st.text_input("LoRA 权重路径", value="outputs/student_lora")

mode = st.radio("选择回答方", options=["Teacher", "Student", "对比"], index=2)
question = st.text_area("请输入 CPA 问题", value="什么是资本成本？")

if st.button("生成回答"):
    teacher_answer = None
    student_answer = None

    if mode in {"Teacher", "对比"}:
        writer = WriterAgent()
        teacher_answer = writer.answer_question(question)

    if mode in {"Student", "对比"}:
        tokenizer, model = load_student(student_base, Path(lora_path))
        student_answer = chat(tokenizer, model, question)

    col1, col2 = st.columns(2)
    if mode == "Teacher":
        col1.subheader("Teacher")
        col1.write(teacher_answer)
    elif mode == "Student":
        col2.subheader("Student")
        col2.write(student_answer)
    else:
        col1.subheader("Teacher")
        col1.write(teacher_answer)
        col2.subheader("Student")
        col2.write(student_answer)

    rating = st.slider("请对 Student 回答打分 (1-5)", 1, 5, 3)
    feedback = st.text_input("改进建议")
    if st.button("保存反馈"):
        log = {
            "question": question,
            "teacher": teacher_answer,
            "student": student_answer,
            "rating": rating,
            "feedback": feedback,
        }
        log_dir = Path("logs")
        log_dir.mkdir(exist_ok=True)
        log_path = log_dir / "feedback.jsonl"
        with log_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(log, ensure_ascii=False) + "\n")
        st.success(f"已写入 {log_path}")
