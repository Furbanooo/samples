from typing_extensions import TypedDict, Annotated
from typing import List
from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages

class overallState(TypedDict):
    messages: Annotated[List[BaseMessage], add_messages]
    next_agent: str
    topic_breakdown: str
    researcher_notes: str
    writer_draft: str