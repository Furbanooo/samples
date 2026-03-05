from dotenv import load_dotenv
from pydantic import BaseModel, Field
from typing import List, TypedDict, Required
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph


load_dotenv() 
llm = ChatOpenAI(model="gpt-4-0613", temperature=0.7)
config = {"configarable": {"thread_id": "research_agent_thread"}}

class SubTopic(BaseModel):
    title: str
    description: str
    subtopics: List["SubTopic"] = Field(default_factory=list)


class TopicBreakdownState(TypedDict, total=False):
    Topic: Required[str]
    estimatedDepth: Required[int]
    topLevelSubtopicCount: int
    initialFocus: str  
    breakdownFeedback: str  
    editorialFeedback: str
    subTopics: List[SubTopic]
    domains: List[SubTopic]
    experts: List["Expert"]
    humanPrompt: str  

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
    return [subtopic for subtopic in subtopics if subtopic.subtopics] # I'm trying to get only the top subtopics as domains, but it's not working as expected. I need to debug this.

def gather_initial_focus(state: TopicBreakdownState):
    Topic = state['Topic']
    return {
        'humanPrompt': f"Do you want to focus on any specific part of '{Topic}' or what's on your mind? (Press enter to skip)"
    }

def breakdown_topic(state: TopicBreakdownState):
    Topic = state['Topic']
    estimatedDepth = state['estimatedDepth']
    editorialFeedback = state.get('editorialFeedback', '')
    initialFocus = state.get('initialFocus', '')
    
    focus_context = f" Focus specifically on: {initialFocus}" if initialFocus else ""

    structured_llm = llm.with_structured_output(TopicTree)

    system_message = SystemMessage(content=topicBreakdown_instructions.format(Topic=Topic, editorialFeedback=editorialFeedback))
    user_message = HumanMessage(
        content=(
            f"Break down the topic '{Topic}' into concrete top-level subtopics with recursive nested subtopics.{focus_context} "
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

# Show breakdown and ask if user is satisfied
def review_breakdown(state: TopicBreakdownState):
    subTopics = state.get('subTopics', [])
    
    def format_subtopics(topics, indent=0):
        result = ""
        for topic in topics:
            result += "  " * indent + f"- {topic.title}: {topic.description}\n"
            if topic.subtopics:
                result += format_subtopics(topic.subtopics, indent + 1)
        return result
    
    breakdown_display = format_subtopics(subTopics)
    
    return {
        'humanPrompt': f"Here's the breakdown:\n\n{breakdown_display}\nAre you satisfied with this breakdown? (yes/no, or provide feedback for changes)"
    }

def should_regenerate(state: TopicBreakdownState):
    """Check if user wants to regenerate the breakdown based on their feedback"""
    feedback = state.get('breakdownFeedback', '').lower().strip()
    
    if feedback in ['yes', 'y', '', 'ok', 'good', 'satisfied']:
        return 'continue'
    else:
        return 'regenerate'

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
        'regenerate': 'breakdown_topic'
    }
)
builder.add_edge('generate_experts', END)

memory = MemorySaver()
graph = builder.compile(
    checkpointer=memory,
    interrupt_after=['gather_initial_focus', 'review_breakdown']  
)

def run_with_human_feedback(topic: str, depth: int = 3, thread_id: str = "research_thread"):
    config = {"configurable": {"thread_id": thread_id}}
    
    initial_state = {
        "Topic": topic,
        "estimatedDepth": depth
    }
    
    print(f"\n{'='*50}")
    print("STARTING RESEARCH AGENT")
    print(f"{'='*50}\n")
    
    for event in graph.stream(initial_state, config, stream_mode="values"):
        if 'humanPrompt' in event:
            print(event['humanPrompt'])
    
    initial_focus = input("\nYour response: ").strip()
    graph.update_state(config, {"initialFocus": initial_focus})
    
    for event in graph.stream(None, config, stream_mode="values"):
        if 'humanPrompt' in event:
            print(f"\n{event['humanPrompt']}")
    
    # Get user's feedback on breakdown
    while True:
        breakdown_feedback = input("\nYour response: ").strip()
        graph.update_state(config, {"breakdownFeedback": breakdown_feedback, "editorialFeedback": breakdown_feedback})
        final_state = None
        for event in graph.stream(None, config, stream_mode="values"):
            final_state = event
            if 'humanPrompt' in event:
                print(f"\n{event['humanPrompt']}")
        
        snapshot = graph.get_state(config)
        if not snapshot.next:
            break
    
    print(f"\n{'='*50}")
    print("RESEARCH COMPLETE!")
    print(f"{'='*50}\n")
    
    return final_state
