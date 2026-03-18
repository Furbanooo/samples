from typing_extensions import TypedDict, Annotated
from typing import List, Optional, Dict, Any, Literal
from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages
from pydantic import BaseModel, Field

# Shared data models — imported by every agent.
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


# Writer shared models
class ReportVisual(BaseModel):
    """
    A visual element to be rendered alongside a report section.
    The writer agent produces the spec; the renderer turns it into HTML/SVG.

    visual_type options:
      - bar_chart      : compare values across categories
      - line_chart     : show trends over time or sequence
      - pie_chart      : show proportions of a whole
      - flow_diagram   : show a process or sequence of steps
      - tree_diagram   : show hierarchy or taxonomy
      - comparison_table: compare items across multiple attributes
      - none           : no visual needed for this section
    """
    visual_type: Literal[
        "bar_chart", "line_chart", "pie_chart",
        "flow_diagram", "tree_diagram", "comparison_table", "none"
    ]
    title:       str                    = Field(..., description="Title shown above the visual.")
    description: str                    = Field(..., description="One sentence explaining what the visual shows.")
    # Structured data the renderer uses to build the visual.
    # Shape depends on visual_type — see writer.py for per-type conventions.
    data:        Dict[str, Any]         = Field(default_factory=dict)


class WrittenSection(BaseModel):
    """One finished section of the report, produced by a write_section node."""
    index:       int            = Field(..., description="Order of this section in the final report.")
    title:       str
    prose:       str            = Field(..., description="Full markdown prose for this section.")
    visual:      ReportVisual   = Field(..., description="Visual spec (type='none' if not needed).")
    key_points:  List[str]      = Field(default_factory=list, description="3-5 bullet takeaways.")

# Shared state
class overallState(TypedDict):
    messages:         Annotated[List[BaseMessage], add_messages]
    next_agent:       str
    topic_breakdown:  Optional[TopicBreakdownResult]
    researcher_notes: Optional[str]
    writer_draft:     Optional[str]