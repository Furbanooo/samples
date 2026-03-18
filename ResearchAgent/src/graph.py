from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver

from .agents.supervisor import supervisor_node
from .agents.expert import expert_node
from .agents.researcher import researcher_node
from .agents.writer import writer_node
from .overallState import overallState


def route_supervisor(state: overallState) -> str:
    return state["next_agent"]


workflow = StateGraph(overallState)

workflow.add_node("supervisor",       supervisor_node)
workflow.add_node("expert_breakdown", expert_node)
workflow.add_node("researcher",       researcher_node)
workflow.add_node("writer",           writer_node)

workflow.set_entry_point("supervisor")
workflow.add_conditional_edges("supervisor", route_supervisor, {
    "expert_breakdown": "expert_breakdown",
    "researcher":       "researcher",
    "writer":           "writer",
    "FINISH":           END,
})

# After each agent finishes, return to supervisor so it can decide the next step
workflow.add_edge("expert_breakdown", "supervisor")
workflow.add_edge("researcher",       "supervisor")
workflow.add_edge("writer",           "supervisor")

checkpointer = MemorySaver()
app = workflow.compile(checkpointer=checkpointer)
