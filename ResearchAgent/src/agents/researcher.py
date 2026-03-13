from dotenv import load_dotenv
from pydantic import BaseModel, Field
from typing import List, TypedDict, Required, Dict, Any
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_community.document_loaders import WikipediaLoader
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph
from overallState import overallState


load_dotenv() 
llm = ChatOpenAI(model="gpt-4-0613", temperature=0.7)
config = {"configarable": {"thread_id": "research_agent_thread"}}

# Define state and data models
tp = overallState.topic_breakdown

class ResearchState(BaseModel):
    search_results: Dict[str, Any] = {}
    deep_questions: List[str] = []
    answers: Dict[str, Any] = {}

class SearchQuery(BaseModel):
    search_query: str = Field(None, description="Search query for retrieval.")


# information   collection for the actual research
search_query_instructions = '''
    Hi {expert.name}, you are an expert in {expert.expertise}. Your task is to help gather information on '{expert.subtopic}' to assist the user in their research and completely understand the {topic}.
    To accomplish this, you can use the following tools:
    1. Tavily Search: Use this tool to search the web for relevant and up-to-date information on the subtopic. Provide specific queries to get the best results.
    2. Wikipedia Loader: Use this tool to retrieve information from Wikipedia on the subtopic. This can provide a good overview and foundational knowledge.
    3. Your own knowledge: As an expert, you can also provide insights and information based on your expertise and experience in the field.
'''

def tavily_search(state: ResearchState, expert: overallState.Expert, topic: str, query: str) -> dict:
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

def wikipedia_search(state: ResearchState, expert: tp.Expert) -> dict:
    wikipedia_loader = WikipediaLoader(expert.subtopic)
    documents = wikipedia_loader.load()
    return {"search_results": {"wikipedia": documents}}

deep_question_generation_instructions = '''
    As an expert in {expert.expertise}, your task is to generate deep and insightful questions about the subtopic '{expert.subtopic}' that would guide a user in their research to gain a comprehensive understanding of the broader topic of {topic}.
    These questions should encourage critical thinking and exploration of the subtopic from multiple angles, including foundational concepts, current trends, controversies, and future directions.
    Consider what a user would need to know to become proficient in this area and what questions would lead them to discover that information.
'''

def generate_deep_questions(state: ResearchState, expert: tp.Expert, topic: str):
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
