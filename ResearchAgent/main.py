from dotenv import load_dotenv
from langchain_core.messages import HumanMessage
from src.graph import app

load_dotenv()

config = {"configurable": {"thread_id": "research_1"}}

print("Research Agent Active")
topic = input("enter the topic: ")

for chunk in app.stream({
    "messages": [HumanMessage(content=f"Research: {topic}")],
}, config, stream_mode="values"):
    if "messages" in chunk:
        msg = chunk["messages"][-1]
        role = msg.type if hasattr(msg, 'type') else msg.__class__.__name__
        print(f"{role.upper()}: {msg.content[:100]}...")