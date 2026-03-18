from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_community.document_loaders import WikipediaLoader
from langchain_core.messages import HumanMessage
from langchain_core.tools import tool
from src.agents.expert import Expert
from src.graph import ResearchState
from prompts import search_query_instructions

tavily = TavilySearchResults(max_results=5)

@tool
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

@tool
def wikipedia_search(state: ResearchState, expert: Expert) -> dict:
    wikipedia_loader = WikipediaLoader(expert.subtopic)
    documents = wikipedia_loader.load()
    return {"search_results": {"wikipedia": documents}}