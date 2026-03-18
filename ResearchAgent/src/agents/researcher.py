import operator
from pydantic import BaseModel, Field
from typing import List, Dict, Any
from typing_extensions import TypedDict, Annotated
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_community.document_loaders import WikipediaLoader
from langgraph.graph import END, START, StateGraph
from langgraph.types import Send

from ..overallState import overallState, Expert, TopicBreakdownResult
from ..prompts import search_query_instructions, deep_question_generation_instructions
from ..models import STRICT, ANALYST

llm        = STRICT   # search queries, deep question generation
llm_deep   = ANALYST  # answer_deep_questions — quality research


class SearchQuery(BaseModel):
    search_query: str = Field(..., description="Search query for retrieval.")

class DeepQuestions(BaseModel):
    deep_questions: List[str] = Field(default_factory=list)


# ---------------------------------------------------------------------------
# State types
#
# Two states are needed:
#
#  ResearchState   — private state for ONE expert's research branch.
#                    Each parallel branch gets its own isolated copy.
#                    Pydantic model so fields are dot-accessible.
#
#  ParallelResearchState — the outer graph state that:
#                    1. receives the list of experts to fan out over, and
#                    2. collects all ExpertResult objects back in after the
#                       parallel branches finish.
#                    TypedDict with an operator.add reducer on expert_results
#                    so parallel writes are merged into one list instead of
#                    overwriting each other.
# ---------------------------------------------------------------------------

class ResearchState(BaseModel):
    """Private state for a single expert's research run."""
    expert:         Expert         = Field(..., description="The expert driving this run.")
    topic:          str            = Field(..., description="The top-level research topic.")
    search_results: Dict[str, Any] = Field(default_factory=dict)
    deep_questions: List[str]      = Field(default_factory=list)
    answers:        Dict[str, str] = Field(default_factory=dict)


class ExpertResult(BaseModel):
    """The finished output of one expert's research branch."""
    expert:   Expert
    answers:  Dict[str, str]
    sources:  Dict[str, Any]


class ParallelResearchState(TypedDict):
    topic:   str
    experts: List[Expert]
    expert_results:   Annotated[List[ExpertResult], operator.add]
    researcher_notes: str


# ---------------------------------------------------------------------------
def tavily_search(state: ResearchState) -> dict:
    expert = state.expert
    topic  = state.topic

    structured_llm = llm.with_structured_output(SearchQuery)
    search_query = structured_llm.invoke([
        SystemMessage(content=search_query_instructions.format(
            expert_name      = expert.name,
            expert_expertise = expert.expertise,
            expert_subtopic  = expert.subtopic,
            topic            = topic,
        )),
        HumanMessage(content=(
            f"Generate a specific search query for '{expert.subtopic}' "
            f"within the broader topic of {topic}."
        )),
    ])

    try:
        tool    = TavilySearchResults()
        results = tool.run(search_query.search_query)
    except Exception:
        results = []   # Tavily is best-effort — don't crash the pipeline
    return {"search_results": {**state.search_results, "web_search": results}}


def wikipedia_search(state: ResearchState) -> dict:
    try:
        loader    = WikipediaLoader(query=state.expert.subtopic)
        documents = loader.load()
        pages     = [doc.page_content for doc in documents]
    except Exception:
        pages = []   # Wikipedia is best-effort — don't crash the pipeline
    return {"search_results": {**state.search_results, "wikipedia": pages}}


def generate_deep_questions(state: ResearchState) -> dict:
    expert = state.expert
    topic  = state.topic

    structured_llm = llm.with_structured_output(DeepQuestions)
    questions = structured_llm.invoke([
        SystemMessage(content=deep_question_generation_instructions.format(
            expert_expertise = expert.expertise,
            expert_subtopic  = expert.subtopic,
            topic            = topic,
        )),
        HumanMessage(content=(
            f"Generate deep questions about '{expert.subtopic}' "
            f"to guide research on {topic}."
        )),
    ])
    return {"deep_questions": questions.deep_questions}


def answer_deep_questions(state: ResearchState) -> dict:
    expert  = state.expert
    answers = dict(state.answers)   # copy — never mutate state in place

    for question in state.deep_questions:
        response = llm_deep.invoke([
            SystemMessage(content=(
                f"As an expert in {expert.expertise}, give a comprehensive "
                f"answer about '{expert.subtopic}'."
            )),
            HumanMessage(content=question),
        ])
        answers[question] = response.content

    return {"answers": answers}


def collect_expert_result(state: ResearchState) -> dict:
    return {
        # This dict is returned to ParallelResearchState.
        # operator.add appends it to the growing list.
        "expert_results": [
            ExpertResult(
                expert  = state.expert,
                answers = state.answers,
                sources = state.search_results,
            )
        ]
    }


expert_branch = StateGraph(ResearchState)
expert_branch.add_node("tavily_search",           tavily_search)
expert_branch.add_node("wikipedia_search",        wikipedia_search)
expert_branch.add_node("generate_deep_questions", generate_deep_questions)
expert_branch.add_node("answer_deep_questions",   answer_deep_questions)
expert_branch.add_node("collect_expert_result",   collect_expert_result)

expert_branch.add_edge(START,                     "tavily_search")
expert_branch.add_edge("tavily_search",           "wikipedia_search")
expert_branch.add_edge("wikipedia_search",        "generate_deep_questions")
expert_branch.add_edge("generate_deep_questions", "answer_deep_questions")
expert_branch.add_edge("answer_deep_questions",   "collect_expert_result")
expert_branch.add_edge("collect_expert_result",   END)

expert_branch_graph = expert_branch.compile()


# ---------------------------------------------------------------------------
# Outer parallel graph
#
# dispatch_research — fans out: returns one Send per expert.
#                     Each Send carries an independent ResearchState so
#                     branches are fully isolated from each other.
#
# gather_results    — runs once all branches have finished.
#                     At this point expert_results already contains every
#                     ExpertResult (merged by operator.add), so this node
#                     just formats the final notes string.
# ---------------------------------------------------------------------------

def dispatch_research(state: ParallelResearchState) -> List[Send]:
    """
    Fan-out: one Send per expert.
    Each Send launches an independent copy of research_expert with its own
    ResearchState — branches run in parallel and share nothing.
    """
    return [
        Send(
            "research_expert",
            ResearchState(expert=expert, topic=state["topic"]),
        )
        for expert in state["experts"]
    ]


def research_expert(state: ResearchState) -> dict:
    """
    Entry point for each parallel branch.
    Runs the full expert_branch_graph and returns the result to the
    outer ParallelResearchState via the expert_results reducer.
    """
    final = None
    for event in expert_branch_graph.stream(state, stream_mode="values"):
        final = event

    return {
        "expert_results": [
            ExpertResult(
                expert  = final['expert'],
                answers = final['answers'],
                sources = final['search_results'],
            )
        ]
    }


def gather_results(state: ParallelResearchState) -> dict:
    """
    Runs once after ALL parallel branches have finished.
    expert_results is already a fully merged list at this point.
    Formats it into a researcher_notes string for overallState.
    """
    notes = []
    for result in state["expert_results"]:
        notes.append(f"\n## {result.expert.name} — {result.expert.subtopic}\n")
        for question, answer in result.answers.items():
            notes.append(f"**Q: {question}**\n{answer}\n")

    return {"researcher_notes": "\n".join(notes)}


parallel_builder = StateGraph(ParallelResearchState)
parallel_builder.add_node("research_expert", research_expert)
parallel_builder.add_node("gather_results",  gather_results)

# dispatch_research is a conditional edge that returns Send objects
# — not a node. add_conditional_edges handles the fan-out.
parallel_builder.add_conditional_edges(
    START,
    dispatch_research,
    # no mapping dict needed when returning Send objects
)
parallel_builder.add_edge("research_expert", "gather_results")
parallel_builder.add_edge("gather_results",  END)

parallel_research_graph = parallel_builder.compile()


def run_parallel_research(breakdown: TopicBreakdownResult) -> str:
    initial_state: ParallelResearchState = {
        "topic":          breakdown.Topic,
        "experts":        breakdown.experts,
        "expert_results": [],
    }

    final = {}
    for event in parallel_research_graph.stream(initial_state, stream_mode="values"):
        final = event

    return final.get("researcher_notes", "")


def researcher_node(state: overallState) -> dict:
    """LangGraph node wrapper: runs all expert research branches in parallel
    and writes the merged notes string back into overallState."""
    notes = run_parallel_research(state["topic_breakdown"])
    return {
        "researcher_notes": notes,
        "messages": [
            AIMessage(
                content=(
                    f"Research complete. Notes gathered from "
                    f"{len(state['topic_breakdown'].experts)} experts."
                ),
                name="Researcher",
            )
        ],
    }