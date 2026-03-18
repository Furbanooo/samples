from pydantic import BaseModel, Field
from typing import Literal
from langchain_core.messages import AIMessage, SystemMessage, HumanMessage
from ..overallState import overallState
from ..prompts import SUPERVISOR_SYSTEM_PROMPT
from ..models import ROUTER

llm = ROUTER


class SupervisorDecision(BaseModel):
    reasoning: str = Field(
        ..., description="One sentence explaining why this agent was chosen."
    )
    next: Literal["expert_breakdown", "researcher", "writer", "FINISH"] = Field(
        ..., description="The next agent to run, or FINISH when complete."
    )


structured_supervisor = llm.with_structured_output(SupervisorDecision)


def supervisor_node(state: overallState) -> dict:
    """
    Reads the last 5 messages from shared state, decides which agent runs
    next, and writes the decision back into overallState.
    """
    decision: SupervisorDecision = structured_supervisor.invoke([
        SystemMessage(content=SUPERVISOR_SYSTEM_PROMPT),
        HumanMessage(content=f"Conversation so far:\n{state['messages'][-5:]}"),
    ])

    return {
        "messages": [
            AIMessage(
                content=(
                    f"[Supervisor] → {decision.next} — {decision.reasoning}"
                ),
                name="Supervisor",
            )
        ],
        "next_agent": decision.next,
    }