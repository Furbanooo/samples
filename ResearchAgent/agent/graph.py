from dotenv import load_dotenv
from pydantic import BaseModel, Field
from typing import List, TypedDict, Required
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI

from langgraph.graph import END, START, StateGraph


load_dotenv() 
llm = ChatOpenAI(model="gpt-4-0613", temperature=0.7)

class SubTopic(BaseModel):
    title: str
    description: str
    subtopics: List["SubTopic"] = Field(default_factory=list)


class TopicBreakdownState(TypedDict, total=False):
    Topic: Required[str]
    estimatedDepth: Required[int]
    topLevelSubtopicCount: int
    editorialFeedback: str
    subTopics: List[SubTopic]
    domains: List[SubTopic]
    experts: List["Expert"]

class TopicTree(BaseModel):
    subTopics: List[SubTopic] = Field(
        default_factory=list,
        description="A list of subtopics, each containing a title, description, and optionally nested subtopics."
    )

class Expert(BaseModel):
    name: str = Field(..., description="The name of the expert.")
    expertise: str= Field(..., description="A brief description of the expert's area of expertise.")
    subtopic: str = Field(..., description="The subtopic that the expert is associated with.")


class ExpertsPayload(BaseModel):
    experts: List[Expert] = Field(default_factory=list)


topicBreakdown_instructions = '''
You are an expert in breaking down complex topics into smaller, more manageable subtopics. 
Rules:

1. Divide {Topic} into concrete top-level subtopics.
2. Any subtopic can contain nested subtopics recursively.
3. Subtopics must be specific, not vague categories.
4. Avoid redundancy across sibling subtopics.
5. Maintain logical learning progression.
6. Avoid over-fragmentation.
7. Do not exceed the specified depth; depth should be dynamic per branch.
8. Some branches can stop earlier if already atomic, while others can go deeper when needed.
9. Use editorial feedback when provided: {editorialFeedback}
10. Output must strictly follow the required structured schema.
11. The structure must be sufficient for someone to reach professional-level understanding.
'''

expert_generation_instructions = ''' 
You are an expert in profiling, your goal is to assign experts to each subtopic. 
Rules:

1. For each provided subtopic, identify a suitable expert.
2. Provide the expert's name, area of expertise, and associated subtopic.
3. Ensure the expert's expertise aligns with the subtopic.
4. Output must strictly follow the required structured schema.
'''

#Breakdown topic into subtopics and defining an expert for each subtopic, and the estimated depth of the topic tree.
def _extract_domains(subtopics: List[SubTopic]) -> List[SubTopic]:
    return [subtopic for subtopic in subtopics if subtopic.subtopics]

def breakdown_topic(state: TopicBreakdownState):
    Topic = state['Topic']
    estimatedDepth = state['estimatedDepth']
    editorialFeedback = state.get('editorialFeedback', '')

    structured_llm = llm.with_structured_output(TopicTree)

    system_message = SystemMessage(content=topicBreakdown_instructions.format(Topic=Topic, editorialFeedback=editorialFeedback))
    user_message = HumanMessage(
        content=(
            f"Break down the topic '{Topic}'  into concrete top-level subtopics. with recursive nested subtopics. "
            f"Keep maximum depth at {estimatedDepth}. "
            "Allow uneven branch depth: stop early for atomic branches and go deeper only when meaningful. "
            "Return the result strictly in the required schema."
        )
    )

    breakdown = structured_llm.invoke([system_message, user_message])

    return {
        'subTopics': breakdown.subTopics,
        'domains': _extract_domains(breakdown.subTopics),
    }

# generate experts
def generate_experts(state: TopicBreakdownState):
    structured_llm = llm.with_structured_output(ExpertsPayload)
    domains= state.get('domains', [])

    system_message = SystemMessage(content=expert_generation_instructions)
    user_message = HumanMessage(
        content=(
            f"Assign experts to the following domains: {domains}. "
            f"Return the result strictly in the required schema."
        )
    )

    experts = structured_llm.invoke([system_message, user_message])

    return {'experts': experts.experts}

builder = StateGraph(TopicBreakdownState)
builder.add_node('breakdown_topic', breakdown_topic)
builder.add_node('generate_experts', generate_experts)
builder.add_edge(START, 'breakdown_topic')
builder.add_edge('breakdown_topic', 'generate_experts')
builder.add_edge('generate_experts', END)

graph = builder.compile()