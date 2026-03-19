from dotenv import load_dotenv
from langchain_openai import ChatOpenAI

load_dotenv()

ROUTER  = ChatOpenAI(model="gpt-4o-mini", temperature=0)
STRICT  = ChatOpenAI(model="gpt-4o-mini", temperature=0,   max_tokens=4096)
ANALYST = ChatOpenAI(model="gpt-4o",      temperature=0.7)
WRITER  = ChatOpenAI(model="gpt-4o",      temperature=0.7)
