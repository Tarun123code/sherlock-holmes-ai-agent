import os
from dataclasses import dataclass
from pathlib import Path
from typing import TypedDict

from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain_core.tools import tool
from langchain_groq import ChatGroq
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, StateGraph


# --------------------------------------------------
# 1. Environment and LLM setup
# --------------------------------------------------

load_dotenv()
MODEL = os.getenv("GROQ_MODEL")


def new_llm():
    return ChatGroq(
        temperature=0.3,
        model=MODEL,
        max_retries=8,
        timeout=60,
    )


# Optional basic LLM test
response = new_llm().invoke(
    "Introduce yourself as Sherlock Holmes in one sentence."
)
# print(response.content)


# --------------------------------------------------
# 2. Sherlock Holmes persona
# --------------------------------------------------

@dataclass
class Persona:
    name: str
    voice: str


persona = Persona(
    name="Sherlock Holmes",
    voice="Clipped, deductive, precise.",
)


def _system(persona: Persona) -> str:
    return (
        f"You are {persona.name}, speaking in the first person.\n"
        f"Voice: {persona.voice}\n\n"
        "Use the source book for facts about yourself, other characters, "
        "and events. Do not invent facts that contradict the book. "
        "If the book does not contain an answer, say so in character.\n\n"
        "Stay in character. Never claim to be an AI. "
        "Keep replies natural and concise unless asked for more."
    )


# --------------------------------------------------
# 3. Load the Sherlock Holmes source book
# --------------------------------------------------

BOOK_PATH = Path(__file__).parent / "data" / "sherlock_book.txt"
BOOK_TEXT = BOOK_PATH.read_text(encoding="utf-8")

passages = [
    paragraph
    for paragraph in BOOK_TEXT.strip().split("\n\n")
    if paragraph.strip()
]


# --------------------------------------------------
# 4. Passage retriever
# --------------------------------------------------

class PassageRetriever:
    def __init__(self, passages):
        self.passages = passages

    def invoke(self, query: str):
        results = []

        for passage in self.passages:
            if query.lower() in passage.lower():
                results.append(passage)

        return results


retriever = PassageRetriever(passages)


# --------------------------------------------------
# 5. Search tool for the agent
# --------------------------------------------------

def make_search_tool(retriever):
    @tool
    def search_book(query: str) -> str:
        """Get passages from the Sherlock Holmes source book."""
        docs = retriever.invoke(query)

        if not docs:
            return "No relevant passages found in the book."

        return "\n\n---\n\n".join(docs)

    return search_book


search_tool = make_search_tool(retriever)


# --------------------------------------------------
# 6. Create and run the agent
# --------------------------------------------------

agent = create_agent(
    new_llm(),
    tools=[search_tool],
    system_prompt=_system(persona),
    checkpointer=InMemorySaver(),
)

result = agent.invoke(
    {
        "messages": [
            ("user", "What can you tell me about Professor Moriarty?")
        ]
    },
    config={
        "configurable": {"thread_id": "sherlock-demo"}
    },
)

draft = result["messages"][-1].content

# Collect passages returned by tool messages in the agent run.
source_passages = "\n\n".join(
    message.content
    for message in result["messages"]
    if getattr(message, "type", None) == "tool"
)


# --------------------------------------------------
# 7. Reflection and revision functions
# --------------------------------------------------

def reflect_in_character(draft, persona, source_passages):
    prompt = f""" You are {persona.name}.
            Source book passages: {source_passages}
            Answer to review:{draft}

            Compare the answer with the source book passages.
            Identify facts that are unsupported or contradict the source.
            Suggest how to improve the answer using only supported facts.
            """

    feedback = new_llm().invoke(prompt)
    return feedback.content


def revise_answer(draft, feedback, persona):
    prompt = f""" You are {persona.name}.
            Your speaking style is: {persona.voice}
            Original answer: {draft}
            Review feedback: {feedback}
            Revise the original answer using the feedback.
            Keep the answer accurate to the source book.
            Return only the revised answer."""

    response = new_llm().invoke(prompt)
    return response.content


# --------------------------------------------------
# 8. Direct reflection/revision run
# --------------------------------------------------

feedback = reflect_in_character(draft, persona, source_passages)
final_answer = revise_answer(draft, feedback, persona)

print("Final answer (direct run):", final_answer)


# --------------------------------------------------
# 9. LangGraph reflection workflow
# --------------------------------------------------

class ReflectionState(TypedDict):
    draft: str
    feedback: str
    needs_revision: bool
    final_answer: str


def reflection_node(state: ReflectionState):
    feedback = reflect_in_character(
        state["draft"],
        persona,
        source_passages,
    )
    needs_revision = "unsupported" in feedback.lower()

    return {
        "feedback": feedback,
        "needs_revision": needs_revision,
    }


def revise_node(state: ReflectionState):
    final_answer = revise_answer(
        state["draft"],
        state["feedback"],
        persona,
    )
    return {"final_answer": final_answer}


def route_after_reflection(state: ReflectionState):
    if state["needs_revision"]:
        return "revise"
    return "end"


builder = StateGraph(ReflectionState)

builder.add_node("reflection", reflection_node)
builder.add_node("revise", revise_node)

builder.add_edge(START, "reflection")
builder.add_conditional_edges(
    "reflection",
    route_after_reflection,
    {
        "revise": "revise",
        "end": END,
    },
)
builder.add_edge("revise", END)

graph = builder.compile()

graph_result = graph.invoke(
    {
        "draft": draft,
        "feedback": "",
        "needs_revision": False,
        "final_answer": "",
    }
)

print("LangGraph result:", graph_result["final_answer"])
