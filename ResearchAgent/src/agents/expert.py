from pydantic import BaseModel, Field
from typing import List, Optional
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph
from ..overallState import overallState, SubTopic, Expert, TopicBreakdownResult
from ..prompts import topicBreakdown_instructions, expert_generation_instructions
from ..models import STRICT

llm = STRICT


class TopicBreakdownSchema(BaseModel):
    """
    Schema fed to the LLM via with_structured_output().
    Kept separate from TopicBreakdownResult so the LLM schema can evolve
    independently from the shared contract in overallState.py.
    """
    Topic:          str
    estimatedDepth: int
    subTopics:      List[SubTopic] = Field(default_factory=list)
    domains:        List[str]      = Field(default_factory=list)
    experts:        List[Expert]   = Field(default_factory=list)

class ExpertsPayload(BaseModel):
    experts: List[Expert] = Field(default_factory=list)

class privateState(TopicBreakdownSchema):
    """
    Extends the LLM schema with fields that are only meaningful inside this
    agent's graph (feedback loops, human prompts, flags, etc.).
    These fields must NEVER leak into overallState directly.
    """
    topLevelSubtopicCount: int = 0
    initialFocus: str = ""
    breakdownFeedback: str = ""
    editorialFeedback: str = ""
    humanPrompt: str = ""

def _extract_domains(subtopics: List[SubTopic]) -> List[str]:
    return [subtopic.title for subtopic in subtopics]

def gather_initial_focus(state: privateState):
    Topic = state.Topic
    return {
        'humanPrompt': (
            f"Do you want to focus on any specific part of '{Topic}' "
            "or should the breakdown cover all aspects broadly? Please specify if you have a preference. \n(press Enter to skip)"
        )
    }


def breakdown_topic(state: privateState):
    Topic = state.Topic
    estimatedDepth = state.estimatedDepth
    editorialFeedback = state.editorialFeedback
    initialFocus = state.initialFocus

    focus_context = f" Focus specifically on: {initialFocus}" if initialFocus else ""

    structured_llm = llm.with_structured_output(TopicBreakdownSchema)

    system_message = SystemMessage(
        content=topicBreakdown_instructions.format(
            Topic=Topic, editorialFeedback=editorialFeedback
        )
    )
    user_message = HumanMessage(
        content=(
            f"Break down the topic '{Topic}' into subtopics.{focus_context} "
            f"Rules: max {estimatedDepth} levels deep, max 5 top-level subtopics, "
            "max 3 children per node. Stop early for atomic subtopics. "
            "Be concise — short titles and one-sentence descriptions only. "
            "Return the result strictly in the required schema."
        )
    )

    breakdown = structured_llm.invoke([system_message, user_message])

    return {
        'subTopics':   breakdown.subTopics,
        'domains':     _extract_domains(breakdown.subTopics),
        'humanPrompt': '',   # clear stale prompt so it doesn't re-print on the next event
    }


def review_breakdown(state: privateState):
    subTopics = state.subTopics

    def format_subtopics(topics, indent=0):
        result = ""
        for topic in topics:
            result += "  " * indent + f"- {topic.title}: {topic.description}\n"
            if topic.subtopics:
                result += format_subtopics(topic.subtopics, indent + 1)
        return result

    breakdown_display = format_subtopics(subTopics)

    return {
        'humanPrompt': (
            f"Here's the breakdown:\n\n{breakdown_display}\n"
            "Are you satisfied with this breakdown? "
            "(yes/no, or provide feedback for changes)"
        )
    }


def should_regenerate(state: privateState) -> str:
    feedback = state.breakdownFeedback.lower().strip()
    if feedback in ['yes', 'y', '', 'ok', 'good', 'satisfied']:
        return 'continue'
    return 'regenerate'


def generate_experts(state: privateState):
    structured_llm = llm.with_structured_output(ExpertsPayload)
    domains = state.domains

    system_message = SystemMessage(content=expert_generation_instructions)
    user_message = HumanMessage(
        content=(
            f"Assign experts to the following domains: {domains}. "
            "Return the result strictly in the required schema."
        )
    )

    experts_payload = structured_llm.invoke([system_message, user_message])
    return {'experts': experts_payload.experts}


builder = StateGraph(privateState)
builder.add_node('gather_initial_focus', gather_initial_focus)
builder.add_node('breakdown_topic', breakdown_topic)
builder.add_node('review_breakdown', review_breakdown)
builder.add_node('generate_experts', generate_experts)

builder.add_edge(START, 'gather_initial_focus')
builder.add_edge('gather_initial_focus', 'breakdown_topic')
builder.add_edge('breakdown_topic', 'review_breakdown')
builder.add_conditional_edges(
    'review_breakdown',
    should_regenerate,
    {
        'continue': 'generate_experts',
        'regenerate': 'breakdown_topic',
    }
)
builder.add_edge('generate_experts', END)

memory = MemorySaver()
graph = builder.compile(
    checkpointer=memory,
    interrupt_after=['gather_initial_focus', 'review_breakdown'],
)


# Runner with human-in-the-loop feedback
def run_with_human_feedback(
    topic: str,
    depth: int = 3,
    thread_id: str = "research_thread",
) -> Optional[dict]:
    config = {"configurable": {"thread_id": thread_id}}

    initial_state = {
        "Topic": topic,
        "estimatedDepth": depth,
    }

    print(f"\n{'='*50}")
    print("STARTING RESEARCH AGENT")
    print(f"{'='*50}\n")
    last_shown = ""

    def _show_prompt(event: dict) -> None:
        nonlocal last_shown
        prompt = event.get('humanPrompt', '')
        if prompt and prompt != last_shown:
            print(f"\n{prompt}")
            last_shown = prompt

    # --- Step 1: gather_initial_focus ---
    for event in graph.stream(initial_state, config, stream_mode="values"):
        _show_prompt(event)

    initial_focus = input("\nYour response: ").strip()
    graph.update_state(config, {"initialFocus": initial_focus})

    # --- Step 2: breakdown_topic → review_breakdown ---
    for event in graph.stream(None, config, stream_mode="values"):
        _show_prompt(event)

    # --- Step 3: iterative feedback loop ---
    final_state = {}
    while True:
        breakdown_feedback = input("\nYour response: ").strip()
        graph.update_state(
            config,
            {"breakdownFeedback": breakdown_feedback, "editorialFeedback": breakdown_feedback},
        )

        for event in graph.stream(None, config, stream_mode="values"):
            final_state = event
            _show_prompt(event)

        snapshot = graph.get_state(config)
        if not snapshot.next:
            break

    return final_state


# Handoff to the shared graph
def build_topic_breakdown_result(final_state: dict) -> TopicBreakdownResult:
    return TopicBreakdownResult(
        Topic=final_state.get('Topic', ''),
        estimatedDepth=final_state.get('estimatedDepth', 0),
        subTopics=final_state.get('subTopics', []),
        domains=final_state.get('domains', []),
        experts=final_state.get('experts', []),
    )

def expert_node(state: overallState) -> dict:
    """LangGraph node wrapper: extracts the topic from state, runs the
    expert breakdown sub-graph with human-in-the-loop, and writes the
    resulting TopicBreakdownResult back into overallState."""
    topic = ""
    for msg in state["messages"]:
        content = getattr(msg, "content", "")
        if content.startswith("Research:"):
            topic = content.removeprefix("Research:").strip()
            break
    if not topic and state["messages"]:
        topic = state["messages"][0].content

    final_state = run_with_human_feedback(topic=topic)
    if final_state is None:
        return {}

    breakdown = build_topic_breakdown_result(final_state)
    return {
        "topic_breakdown": breakdown,
        "messages": [
            AIMessage(
                content=(
                    f"Expert breakdown complete for '{topic}'. "
                    f"Found {len(breakdown.experts)} experts across "
                    f"{len(breakdown.domains)} domains."
                ),
                name="Expert",
            )
        ],
    }