from dotenv import load_dotenv
from pydantic import BaseModel, Field
from typing import List, TypedDict, Required, Dict, Any
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_community.document_loaders import WikipediaLoader
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph


load_dotenv() 
llm = ChatOpenAI(model="gpt-4-0613", temperature=0.7)
config = {"configarable": {"thread_id": "research_agent_thread"}}