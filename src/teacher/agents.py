import json
import os
from dataclasses import dataclass
from typing import List, Optional, Sequence

from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI


DEEPSEEK_MODEL = os.getenv("DEEPSEEK_MODEL", "deepseek-chat")
DEEPSEEK_API_BASE = os.getenv("DEEPSEEK_API_BASE")
DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY")


def _build_client(temperature: float = 0.2) -> Optional[ChatOpenAI]:
    """Instantiate a ChatOpenAI client if credentials are available."""
    if not DEEPSEEK_API_KEY or not DEEPSEEK_API_BASE:
        return None
    return ChatOpenAI(
        model=DEEPSEEK_MODEL,
        temperature=temperature,
        api_key=DEEPSEEK_API_KEY,
        base_url=DEEPSEEK_API_BASE,
    )


def _fallback_response(prompt: str, tag: str) -> str:
    """Offline-safe fallback message used when API credentials are missing."""
    return f"[{tag}] {prompt[:120]} ... (请配置 DEEPSEEK_API_KEY 和 DEEPSEEK_API_BASE 以获得真实生成内容)"


@dataclass
class OutlineNode:
    section: str
    bullet_points: List[str]


@dataclass
class QAItem:
    id: int
    topic: str
    difficulty: str
    input: str
    output: str
    teaching_note: Optional[str] = None
    review: Optional[str] = None
    score: Optional[float] = None


class PlannerAgent:
    """Breaks a topic into teachable outline nodes."""

    def __init__(self, client: Optional[ChatOpenAI] = None) -> None:
        self.client = client or _build_client()
        self.prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    "你是注册会计师课程的教案设计师，请输出章节大纲，每节给出 2-4 个要点，使用 JSON 数组，每项含 section 和 bullet_points。",
                ),
                ("human", "科目: {topic}"),
            ]
        )

    def plan(self, topic: str) -> List[OutlineNode]:
        rendered = self.prompt.format(topic=topic)
        if not self.client:
            fallback = _fallback_response(rendered, "Planner")
            return [OutlineNode(section=topic, bullet_points=[fallback])]
        response = self.client.invoke(rendered)
        content = response.content.strip()
        try:
            data = json.loads(content)
            return [OutlineNode(**item) for item in data]
        except Exception:
            return [OutlineNode(section=topic, bullet_points=[content])]


class WriterAgent:
    """Produces teaching notes or QA pairs for outline nodes."""

    def __init__(self, client: Optional[ChatOpenAI] = None) -> None:
        self.client = client or _build_client(temperature=0.7)
        self.qa_prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    "你是 CPA 讲解老师，输出简短单轮 QA：\n- 题型：填空/计算/简答，题干<=30字，答案<=40字，可含1条公式或1-2步计算；\n- 只出一问一答，不要解析/点评/多问多答；\n- 题干紧扣给定要点，不扩展新话题；\n- 输出格式严格为：问题：...\\n答案：...",
                ),
                (
                    "human",
                    "科目: {topic}\n要点: {bullets}\n难度: {difficulty}\n变体序号: {variant}",
                ),
            ]
        )
        self.note_prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    "用 80-120 字写 CPA 知识点讲解，突出公式/定义，避免冗长案例。",
                ),
                (
                    "human",
                    "标题: {heading}\n要点: {bullets}\n请输出讲解。",
                ),
            ]
        )
        self.answer_prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    "你是 CPA 教学专家，回答要简短、有条理，避免长段落。",
                ),
                ("human", "问题: {question}"),
            ]
        )

    def generate_note(self, heading: str, bullets: Sequence[str]) -> str:
        rendered = self.note_prompt.format(heading=heading, bullets="; ".join(bullets))
        if not self.client:
            return _fallback_response(rendered, "TeachingNote")
        return self.client.invoke(rendered).content.strip()

    def generate_qa(
        self, topic: str, bullets: Sequence[str], difficulty: str, variant: int = 1
    ) -> tuple[str, str]:
        rendered = self.qa_prompt.format(
            topic=topic,
            bullets="; ".join(bullets),
            difficulty=difficulty,
            variant=variant,
        )
        if not self.client:
            placeholder = _fallback_response(rendered, "QA")
            return placeholder, placeholder
        reply = self.client.invoke(rendered).content.strip()
        if "答案：" in reply:
            parts = reply.split("答案：", maxsplit=1)
            return parts[0].replace("问题：", "").strip(), parts[1].strip()
        if "答：" in reply:
            parts = reply.split("答：", maxsplit=1)
            return parts[0].replace("问：", "").strip(), parts[1].strip()
        return reply, ""

    def answer_question(self, question: str) -> str:
        rendered = self.answer_prompt.format(question=question)
        if not self.client:
            return _fallback_response(rendered, "TeacherAnswer")
        return self.client.invoke(rendered).content.strip()


class ReviewerAgent:
    """Scores QA quality and suggests fixes."""

    def __init__(self, client: Optional[ChatOpenAI] = None) -> None:
        self.client = client or _build_client()
        self.prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    "你是注册会计师出题质检专家，只返回 JSON，不要代码块，不要多余文字。键：score(0-10), review(20字内指出是否贴合要点、是否简洁)。",
                ),
                (
                    "human",
                    "问题: {question}\n答案: {answer}\n请直接输出 JSON，例如 {{\"score\": 8.5, \"review\": \"答案简短但缺少公式\"}}",
                ),
            ]
        )

    def review(self, question: str, answer: str) -> tuple[float, str]:
        rendered = self.prompt.format(question=question, answer=answer)
        if not self.client:
            return 5.0, _fallback_response(rendered, "Review")
        content = self.client.invoke(rendered).content.strip()
        cleaned = content.strip()
        if cleaned.startswith("```"):
            cleaned = cleaned.strip("`").strip()
            if cleaned.lower().startswith("json"):
                cleaned = cleaned[4:].strip()
        try:
            data = json.loads(cleaned)
            return float(data.get("score", 0)), data.get("review", "")
        except Exception:
            return 0.0, content


class DatasetSynthesizer:
    """High-level orchestrator building a list of QAItem objects."""

    def __init__(self, planner: PlannerAgent, writer: WriterAgent, reviewer: ReviewerAgent) -> None:
        self.planner = planner
        self.writer = writer
        self.reviewer = reviewer

    def build(
        self,
        topic: str,
        num_questions: int = 5,
        difficulties: Sequence[str] | None = None,
        min_score: float | None = None,
        max_attempts: int | None = None,
        use_outline: bool = False,
        flush_every: int | None = None,
        flush_callback: Optional[callable] = None,
        start_id: int = 1,
    ) -> List[QAItem]:
        outline = self.planner.plan(topic) if use_outline else [OutlineNode(section=topic, bullet_points=[topic])]
        difficulties = list(difficulties or ["easy", "medium", "hard"])
        attempt_cap = max_attempts or num_questions * 5

        expanded_outline: list[tuple[str, str, str]] = []
        for node in outline:
            bullets = node.bullet_points or [node.section]
            # 跳过讲义生成以加速；教学笔记留空字符串
            note = "" if not use_outline else self.writer.generate_note(node.section, bullets)
            for bullet in bullets:
                expanded_outline.append((node.section, note, bullet))

        dataset: list[QAItem] = []
        seen_inputs: set[str] = set()
        chunk_buffer: list[QAItem] = []
        attempts = 0
        total_slots = len(expanded_outline)
        while len(dataset) < num_questions and attempts < attempt_cap:
            section, note, bullet = expanded_outline[attempts % total_slots]
            variant = attempts // total_slots + 1
            difficulty = difficulties[attempts % len(difficulties)]
            question, answer = self.writer.generate_qa(
                topic=section,
                bullets=[bullet],
                difficulty=difficulty,
                variant=variant,
            )
            score, review = self.reviewer.review(question, answer)
            attempts += 1
            if min_score is not None and score < min_score:
                continue
            if question in seen_inputs:
                continue
            seen_inputs.add(question)
            item = QAItem(
                id=start_id + len(dataset),
                topic=section,
                difficulty=difficulty,
                input=question,
                output=answer,
                teaching_note=note,
                review=review,
                score=score,
            )
            dataset.append(item)
            chunk_buffer.append(item)
            if flush_every and flush_callback and len(chunk_buffer) >= flush_every:
                flush_callback(chunk_buffer)
                chunk_buffer = []

        if chunk_buffer and flush_callback:
            flush_callback(chunk_buffer)

        return dataset

    @staticmethod
    def to_jsonl(records: Sequence[QAItem]) -> str:
        return "\n".join(json.dumps(record.__dict__, ensure_ascii=False) for record in records)
