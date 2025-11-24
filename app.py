import json
import os
from pathlib import Path

import streamlit as st

from src.teacher.agents import DatasetSynthesizer, PlannerAgent, ReviewerAgent, WriterAgent
from src.student.inference import chat, load_student

st.set_page_config(page_title="CPA Teacher vs Student", page_icon="📚", layout="wide")

st.markdown(
    """
    <style>
    .hero {
        padding: 1.5rem;
        border-radius: 14px;
        background: linear-gradient(135deg, #f0f4ff 0%, #fef6ed 100%);
        border: 1px solid #e4e7ef;
        margin-bottom: 1rem;
    }
    .pill {
        display: inline-block;
        padding: 0.35rem 0.8rem;
        margin-right: 0.5rem;
        border-radius: 999px;
        background: #f6f8fb;
        border: 1px solid #e4e7ef;
        font-size: 0.85rem;
    }
    .card {
        padding: 1rem;
        border-radius: 12px;
        border: 1px solid #e4e7ef;
        background: #fff;
        min-height: 180px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    """
    <div class="hero">
        <h1 style="margin-bottom:0.5rem;">CPA Teacher vs Student 对比</h1>
        <div class="pill">多智能体教师</div>
        <div class="pill">学生 LoRA 微调</div>
        <p style="margin-top:0.8rem; color:#4c4f69;">快速体验出题-讲解-评分链路，并批量合成教学样本。</p>
    </div>
    """,
    unsafe_allow_html=True,
)

with st.sidebar:
    st.header("模型配置")
    deepseek_ready = bool(os.getenv("DEEPSEEK_API_KEY") and os.getenv("DEEPSEEK_API_BASE"))
    st.write("✅ Teacher API 已配置" if deepseek_ready else "⚠️ Teacher API 未配置，使用占位输出")
    student_base = st.text_input("Student 基座模型", value="Qwen1.5-1.8B-Chat")
    lora_path = st.text_input("LoRA 权重路径", value="outputs/student_lora")
    st.divider()
    st.caption("提示：可在主界面使用批量合成获得 200+ 条训练样本")

mode = st.radio("选择回答方", options=["Teacher", "Student", "对比"], index=2, horizontal=True)
question = st.text_area("请输入 CPA 问题", value="什么是资本成本？", height=120)

col_action, col_meta = st.columns([2, 1])
with col_action:
    if st.button("生成回答", type="primary"):
        teacher_answer = None
        student_answer = None

        if mode in {"Teacher", "对比"}:
            writer = WriterAgent()
            with st.spinner("Teacher 正在回答..."):
                teacher_answer = writer.answer_question(question)

        if mode in {"Student", "对比"}:
            with st.spinner("Student 正在加载权重并作答..."):
                tokenizer, model = load_student(student_base, Path(lora_path))
                student_answer = chat(tokenizer, model, question)

        tabs = st.tabs(["Teacher", "Student"] if mode == "对比" else [mode])
        if mode == "Teacher":
            with tabs[0]:
                st.markdown("<div class='card'>" + (teacher_answer or "暂无回答") + "</div>", unsafe_allow_html=True)
        elif mode == "Student":
            with tabs[0]:
                st.markdown("<div class='card'>" + (student_answer or "暂无回答") + "</div>", unsafe_allow_html=True)
        else:
            with tabs[0]:
                st.markdown("<div class='card'>" + (teacher_answer or "暂无回答") + "</div>", unsafe_allow_html=True)
            with tabs[1]:
                st.markdown("<div class='card'>" + (student_answer or "暂无回答") + "</div>", unsafe_allow_html=True)

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

with col_meta:
    st.markdown("### 快速提示")
    st.write("- Teacher 使用 DeepSeek 接口生成高质量答案")
    st.write("- Student 会加载 LoRA 进行对照")
    st.write("- 下方可一键批量生成训练样本")

st.markdown("---")

with st.expander("🚀 批量合成教学样本", expanded=True):
    bulk_topic = st.text_input("合成主题", value="财务成本管理")
    bulk_num = st.number_input("合成数量", min_value=20, max_value=2000, step=20, value=200)
    col_left, col_right = st.columns([2, 1])
    with col_left:
        st.caption("提示：多智能体会循环大纲要点，确保达到设定数量。")
    with col_right:
        if st.button("开始合成", type="secondary"):
            planner = PlannerAgent()
            writer = WriterAgent()
            reviewer = ReviewerAgent()
            synth = DatasetSynthesizer(planner, writer, reviewer)
            with st.spinner("正在生成教学样本..."):
                dataset = synth.build(topic=bulk_topic, num_questions=int(bulk_num))
            jsonl_text = DatasetSynthesizer.to_jsonl(dataset)
            st.success(f"已生成 {len(dataset)} 条样本，可直接下载或训练。")
            st.download_button(
                "下载 JSONL",
                data=jsonl_text,
                file_name=f"{bulk_topic}_teacher_dataset.jsonl",
                mime="application/json",
            )
