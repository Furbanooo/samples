# Research Agent — A LangGraph Deep Dive

A revisited and expanded version of the
[LangChain Academy Module 4 research assistant](https://github.com/langchain-ai/langchain-academy/blob/main/module-4/research-assistant.ipynb).

The original notebook is a single flat graph with one researcher and one writer node.
This version breaks the system into **four specialised sub-graphs** orchestrated by a
supervisor, adds a full **human-in-the-loop** expert-breakdown stage, and runs research
branches **in parallel** — one branch per expert.

---

## What was revisited

| Aspect | Original (Academy) | This version |
|---|---|---|
| Architecture | Single flat graph | Supervisor + 3 nested sub-graphs |
| Research | One LLM call | Parallel branches per expert (fan-out with `Send`) |
| Expert assignment | Hardcoded personas | LLM-generated experts from topic breakdown |
| Human feedback | None | Interactive breakdown review loop |
| State design | One shared state | Outer `overallState` + private state per sub-graph |
| Report output | Plain text | Full HTML with Chart.js visuals and tables |

---

## System architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        OUTER GRAPH                              │
│                     (overallState)                              │
│                                                                 │
│   ┌─────────┐     ┌──────────────────┐                         │
│   │  START  │────▶│    Supervisor    │                         │
│   └─────────┘     │  (gpt-4o, T=0)  │                         │
│                   │                  │                         │
│                   │  reads last 5    │                         │
│                   │  messages and    │                         │
│                   │  decides next    │                         │
│                   └────────┬─────────┘                         │
│           ┌────────────────┼──────────────────┐                │
│           ▼                ▼                  ▼                │
│   ┌───────────────┐ ┌────────────┐ ┌────────────────┐         │
│   │ expert_break- │ │ researcher │ │     writer     │         │
│   │     down      │ │            │ │                │         │
│   └───────┬───────┘ └─────┬──────┘ └───────┬────────┘         │
│           │               │                │                   │
│           └───────────────┴────────────────┘                   │
│                           │ (all loop back to Supervisor)      │
│                           ▼                                    │
│                         END                                    │
└─────────────────────────────────────────────────────────────────┘
```

---

## Sub-graph 1 — Expert Breakdown (human-in-the-loop)

The first sub-graph uses `interrupt_after` to pause and collect human input at two points.

```
┌─────────────────────────────────────────────────────┐
│               EXPERT SUB-GRAPH                      │
│                 (privateState)                      │
│                                                     │
│  [START]                                            │
│     │                                               │
│     ▼                                               │
│  gather_initial_focus  ◀── interrupt here           │
│     │                       ↕ human types focus     │
│     ▼                                               │
│  breakdown_topic  ──── LLM breaks topic into        │
│     │                  subtopics + domains          │
│     ▼                                               │
│  review_breakdown  ◀── interrupt here               │
│     │                   ↕ human reviews & gives     │
│     │                     feedback                  │
│     ▼                                               │
│  should_regenerate?                                 │
│     │                                               │
│     ├── "no / feedback" ──▶ breakdown_topic (retry) │
│     │                                               │
│     └── "yes" ──▶ generate_experts                  │
│                        │                            │
│                       END                           │
└─────────────────────────────────────────────────────┘
```

**Key concepts used:**
- `interrupt_after` — pauses graph, waits for human
- `checkpointer` + `thread_id` — saves state between pauses
- `graph.update_state()` — injects human input into frozen state
- `graph.stream(None, config)` — resumes from checkpoint

---

## Sub-graph 2 — Parallel Researcher (fan-out with Send)

One branch per expert, all running in parallel. Results are merged back via a reducer.

```
┌─────────────────────────────────────────────────────────────────┐
│                   RESEARCHER SUB-GRAPH                          │
│                  (ParallelResearchState)                        │
│                                                                 │
│  [START]                                                        │
│     │                                                           │
│     ▼                                                           │
│  dispatch_research  ──── returns one Send per expert            │
│     │                                                           │
│     ├──▶ research_expert (Expert A)  ──────────────────┐        │
│     │       └─ inner expert_branch_graph:              │        │
│     │            tavily_search                         │        │
│     │            wikipedia_search                      │        │
│     │            generate_deep_questions               │        │
│     │            answer_deep_questions                 │        │
│     │                                                  │        │
│     ├──▶ research_expert (Expert B)  ──────────────────┤        │
│     │                                                  │ merge  │
│     └──▶ research_expert (Expert C)  ──────────────────┘        │
│                                            │                    │
│                                            ▼                    │
│                                     gather_results              │
│                                     (format notes)             │
│                                            │                    │
│                                           END                   │
└─────────────────────────────────────────────────────────────────┘
```

**Key concepts used:**
- `Send` — fans out to N parallel branches with independent private state
- `Annotated[List[ExpertResult], operator.add]` — reducer merges parallel results
- Nested compiled graph — `expert_branch_graph` runs inside each `research_expert` node

---

## Sub-graph 3 — Writer (parallel section writing)

Report sections are written in parallel, then assembled into a single HTML document.

```
┌───────────────────────────────────────────────────────────────┐
│                     WRITER SUB-GRAPH                          │
│                      (WriterState)                            │
│                                                               │
│  [START]                                                      │
│     │                                                         │
│     ▼                                                         │
│  plan_report  ──── LLM produces structured outline           │
│     │               (4-8 sections with visual specs)         │
│     │                                                         │
│     ▼  dispatch_sections (returns Send per section)          │
│     │                                                         │
│     ├──▶ write_section (Section 0: Introduction)  ──────┐    │
│     ├──▶ write_section (Section 1)  ─────────────────────┤    │
│     ├──▶ write_section (Section 2)  ─────────────────────┤ merge
│     └──▶ write_section (Section N: Conclusion)  ─────────┘    │
│                                          │                    │
│                                          ▼                    │
│                                   assemble_report             │
│                                   (sort + render HTML)        │
│                                          │                    │
│                                         END                   │
└───────────────────────────────────────────────────────────────┘
```

**Key concepts used:**
- Private state per branch (`SectionWritingState`) separate from outer `WriterState`
- `operator.add` reducer collects `WrittenSection` objects from all parallel branches
- `_slice_notes_for_section` — each branch only receives the notes relevant to its section

---

## State design

```
overallState (shared, TypedDict)
├── messages          — full conversation history (add_messages reducer)
├── next_agent        — supervisor's routing decision
├── topic_breakdown   — output of expert sub-graph
├── researcher_notes  — output of researcher sub-graph
└── writer_draft      — final HTML report

privateState (expert sub-graph, Pydantic BaseModel)
├── Topic, estimatedDepth, subTopics, domains, experts
├── initialFocus, editorialFeedback, breakdownFeedback
└── humanPrompt       — cleared after each breakdown_topic run

ResearchState (per expert branch, Pydantic BaseModel)
├── expert, topic
├── search_results    — merged web + Wikipedia results
├── deep_questions
└── answers

WriterState (writer sub-graph, TypedDict)
├── topic, researcher_notes
├── outline           — list of PlannedSection
├── written_sections  — operator.add reducer
└── writer_draft
```

---

## Running it

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# set keys in .env
OPENAI_API_KEY=...
TAVILY_API_KEY=...

python main.py
```

---

## Key LangGraph concepts exercised

| Concept | Where |
|---|---|
| `StateGraph` + TypedDict state | `overallState`, `WriterState`, `ParallelResearchState` |
| `StateGraph` + Pydantic state | `privateState`, `ResearchState`, `SectionWritingState` |
| `add_messages` reducer | `overallState.messages` |
| `operator.add` reducer | parallel branch results |
| Supervisor routing loop | `graph.py` |
| `interrupt_after` + human-in-the-loop | `expert.py` |
| `Send` fan-out (parallel branches) | `researcher.py`, `writer.py` |
| Nested compiled sub-graphs | all three agent files |
| `MemorySaver` checkpointer | expert sub-graph, outer graph |
| `with_structured_output` | every LLM call |
