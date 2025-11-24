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

    return f"[{tag}] {prompt[:120]} ... (请配置 DEEPSEEK_API_KEY 与 DEEPSEEK_API_BASE 以获得真实生成内容)"


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
                    "你是注册会计师课程的教案设计师，请输出分节大纲，每节附 2-4 个要点。",
                ),
                (
                    "human",
                    "科目: {topic}\n请以 JSON 数组输出，字段 section 与 bullet_points",
                ),
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
                    "你是 CPA 讲解老师，根据大纲要点生成问答题（中文），并附标准答案，确保不同变体的提问方式各有侧重。",
                ),
                (
                    "human",
                    "科目: {topic}\n要点: {bullets}\n难度: {difficulty}\n变体序号: {variant}\n请输出问题和标准答案，避免与同主题已有题目重复。",
                ),
            ]
        )
        self.note_prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    "为 CPA 知识点写简短 teaching_note，便于学生快速理解。",
                ),
                (
                    "human",
                    "标题: {heading}\n要点: {bullets}\n请输出 80-120 字讲解。",
                ),
            ]
        )
        self.answer_prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    "你是 CPA 教学专家，直接回答学生的提问，保持简洁有条理。",
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
        # simple split heuristic
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
                    "你是注册会计师出题专家，请对以下问答进行评分与点评，满分 10 分。",
                ),
                (
                    "human",
                    "问题: {question}\n答案: {answer}\n请输出 JSON，包含 score 与 review。",
                ),
            ]
        )

    def review(self, question: str, answer: str) -> tuple[float, str]:
        rendered = self.prompt.format(question=question, answer=answer)
        if not self.client:
            return 5.0, _fallback_response(rendered, "Review")
        content = self.client.invoke(rendered).content.strip()
        try:
            data = json.loads(content)
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
    ) -> List[QAItem]:
        outline = self.planner.plan(topic)
        difficulties = list(difficulties or ["easy", "medium", "hard"])

        # Gather all outline bullets to allow round-robin generation even if
        # the outline is short, preventing early termination when num_questions
        # is large (e.g., 200+).
        expanded_outline: list[tuple[str, str, str]] = []  # (section, note, bullet)
        for node in outline:
            bullets = node.bullet_points or [node.section]
            note = self.writer.generate_note(node.section, bullets)
            for bullet in bullets:
                expanded_outline.append((node.section, note, bullet))

        if not expanded_outline:
            expanded_outline.append((topic, self.writer.generate_note(topic, [topic]), topic))

        dataset: list[QAItem] = []
        total_slots = len(expanded_outline)
        while len(dataset) < num_questions:
            section, note, bullet = expanded_outline[len(dataset) % total_slots]
            variant = len(dataset) // total_slots + 1
            difficulty = difficulties[len(dataset) % len(difficulties)]
            question, answer = self.writer.generate_qa(
                topic=section,
                bullets=[bullet],
                difficulty=difficulty,
                variant=variant,
            )
            score, review = self.reviewer.review(question, answer)
            dataset.append(
                QAItem(
                    id=len(dataset) + 1,
                    topic=section,
                    difficulty=difficulty,
                    input=question,
                    output=answer,
                    teaching_note=note,
                    review=review,
                    score=score,
                )
            )
        return dataset

    @staticmethod
    def to_jsonl(records: Sequence[QAItem]) -> str:
        return "\n".join(json.dumps(record.__dict__, ensure_ascii=False) for record in records)
