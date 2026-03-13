from dotenv import load_dotenv
from langchain_core.messages import AIMessage
from langchain_openai import ChatOpenAI
from src.prompts import SUPERVISOR_instructions
from typing import Dict, Any

load_dotenv()
llm = ChatOpenAI(model="gpt-4-0613", temperature=0.7)

supervisor = llm.with_structured_output(SUPERVISOR_instructions)

def supervisor_node(state: Dict[str, Any]) -> Dict[str, Any]:
    result = supervisor.invoke({"messages": state["messages"][-5:]})
    agent = result.content.strip().lower()
    return {
        "messages": [AIMessage(content=f"Routing to: {agent}", name="Supervisor")],
        "next_agent": agent
    }