from typing_extensions import TypedDict, Annotated
from typing import List, Optional
from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages
from pydantic import BaseModel, Field

# Shared data models
class SubTopic(BaseModel):
    title: str
    description: str
    subtopics: List["SubTopic"] = Field(default_factory=list)

SubTopic.model_rebuild()


class Expert(BaseModel):
    name:      str = Field(..., description="The name of the expert.")
    expertise: str = Field(..., description="The expert's area of expertise.")
    subtopic:  str = Field(..., description="The subtopic this expert covers.")


class TopicBreakdownResult(BaseModel):
    """Clean output of expert_graph — written into overallState.topic_breakdown."""
    Topic:          str
    estimatedDepth: int
    subTopics:      List[SubTopic]
    domains:        List[str]
    experts:        List[Expert]


# Shared state
class overallState(TypedDict):
    messages:         Annotated[List[BaseMessage], add_messages]
    next_agent:       str
    topic_breakdown:  Optional[TopicBreakdownResult]
    researcher_notes: Optional[str]
    writer_draft:     Optional[str]