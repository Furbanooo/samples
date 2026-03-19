import os
import webbrowser
import tempfile
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage
from src.graph import app

load_dotenv()

config = {"configurable": {"thread_id": "research_1"}}

print("Research Agent Active")
topic = input("enter the topic: ")

final_chunk = {}
for chunk in app.stream({
    "messages": [HumanMessage(content=f"Research: {topic}")],
}, config, stream_mode="values"):
    final_chunk = chunk
    if "messages" in chunk:
        msg = chunk["messages"][-1]
        role = msg.type if hasattr(msg, 'type') else msg.__class__.__name__
        print(f"{role.upper()}: {msg.content[:100]}...")

# ── Terminal summary ──────────────────────────────────────────────────────────
report = final_chunk.get("writer_draft", "")
if report:
    print(f"\n{'='*60}")
    print("REPORT PREVIEW (first 2000 chars)")
    print('='*60)
    print(report[:2000])
    if len(report) > 2000:
        print(f"\n... [{len(report) - 2000} more chars]")
    print(f"\n{'='*60}\n")

    # ── Open in browser ───────────────────────────────────────────────────────
    safe_name = topic.replace(" ", "_")[:40]
    out_path   = os.path.join(os.path.dirname(__file__), f"report_{safe_name}.html")
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(report)
    print(f"Report saved → {out_path}")
    webbrowser.open(f"file://{os.path.abspath(out_path)}")
else:
    print("\n[No report generated — writer_draft was empty]")