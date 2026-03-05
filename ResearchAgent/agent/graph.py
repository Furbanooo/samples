from dotenv import load_dotenv
from pydantic import BaseModel, Field
from typing import List, TypedDict, Required, Dict, Any
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI, tools
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_community.document_loaders import WikipediaLoader
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph


load_dotenv() 
llm = ChatOpenAI(model="gpt-4-0613", temperature=0.7)
config = {"configarable": {"thread_id": "research_agent_thread"}}

# Define state and data models
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

class ResearchState(BaseModel):
    final_state: TopicBreakdownState = Field(..., description="The final state of the topic breakdown process, including the topic, subtopics, and assigned experts.")
    search_results: Dict[str, Any] = Field(default_factory=dict, description="A dictionary containing search results and information gathered for each subtopic, organized by expert and source.")
    deep_questions: List[str] = Field(default_factory=list, description="A list of deep questions generated for each subtopic to guide further research.")
    answers: dict = Field(default_factory=dict, description="A dictionary containing answers to the deep questions, organized by subtopic and source."    
    )

class SearchQuery(BaseModel):
    search_query: str = Field(None, description="Search query for retrieval.")

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

# information   collection for the actual research
search_query_instructions = '''
    Hi {expert.name}, you are an expert in {expert.expertise}. Your task is to help gather information on '{expert.subtopic}' to assist the user in their research and completely understand the {topic}.
    To accomplish this, you can use the following tools:
    1. Tavily Search: Use this tool to search the web for relevant and up-to-date information on the subtopic. Provide specific queries to get the best results.
    2. Wikipedia Loader: Use this tool to retrieve information from Wikipedia on the subtopic. This can provide a good overview and foundational knowledge.
    3. Your own knowledge: As an expert, you can also provide insights and information based on your expertise and experience in the field.
'''
@tools 
def tavily_search(state: ResearchState, expert: Expert, topic: str, query: str) -> dict:
    travily_search = TavilySearchResults()

    # Perform the search using the provided query
    structured_llm = llm.with_structured_output(search_query_instructions)
    search_query = structured_llm.invoke([
        SystemMessage(content=search_query_instructions.format(expert=expert, topic=topic)),
        HumanMessage(content=f"Based on your expertise in {expert.expertise}, generate a specific search query to find relevant information on '{expert.subtopic}' that would help the user understand the broader topic of {topic}.")
    ])

    # Execute the search with the generated query
    results = travily_search.run(search_query.search_query)
    return {"search_results": {"web_search": results}}

def wikipedia_search(state: ResearchState, expert: Expert) -> dict:
    wikipedia_loader = WikipediaLoader(expert.subtopic)
    documents = wikipedia_loader.load()
    return {"search_results": {"wikipedia": documents}}

deep_question_generation_instructions = '''
    As an expert in {expert.expertise}, your task is to generate deep and insightful questions about the subtopic '{expert.subtopic}' that would guide a user in their research to gain a comprehensive understanding of the broader topic of {topic}.
    These questions should encourage critical thinking and exploration of the subtopic from multiple angles, including foundational concepts, current trends, controversies, and future directions.
    Consider what a user would need to know to become proficient in this area and what questions would lead them to discover that information.
'''

def generate_deep_questions(state: ResearchState, expert: Expert, topic: str):
    deep_question=state.deep_questions
    structured_llm = llm.with_structured_output(deep_question)
    questions = structured_llm.invoke([
        SystemMessage(content=deep_question_generation_instructions.format(expert=expert, topic=topic)),
        HumanMessage(content=f"Based on your expertise in {expert.expertise}, generate a list of deep and insightful questions about the subtopic '{expert.subtopic}' that would guide a user in their research to gain a comprehensive understanding of the broader topic of {topic}.")
    ])

    return {"deep_questions": questions.deep_questions}

def answer_deep_questions(state: ResearchState, expert: Expert):
    answers = state.answers
    structured_llm = llm.with_structured_output(answers)
    for question in state.deep_questions:
        answer = structured_llm.invoke([
            SystemMessage(content=f"As an expert in {expert.expertise}, provide a comprehensive and insightful answer to the following question about the subtopic '{expert.subtopic}': {question}")
        ])
        answers[question] = answer

    return {"answers": answers}

# research graph
research_builder = StateGraph(ResearchState)
research_builder.add_node('tavily_search', tavily_search)
research_builder.add_node('wikipedia_search', wikipedia_search)
research_builder.add_node('generate_deep_questions', generate_deep_questions)
research_builder.add_node('answer_deep_questions', answer_deep_questions)

research_builder.add_edge(START, 'tavily_search')
research_builder.add_edge('tavily_search', 'wikipedia_search')
research_builder.add_edge('wikipedia_search', 'generate_deep_questions')
research_builder.add_edge('generate_deep_questions', 'answer_deep_questions')
research_builder.add_edge('answer_deep_questions', END)

