import operator
import json
from dotenv import load_dotenv
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any, Literal
from typing_extensions import TypedDict, Annotated
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from langgraph.graph import END, START, StateGraph
from langgraph.types import Send

from ..overallState import (
    overallState, TopicBreakdownResult,
    ReportVisual, WrittenSection,
)
from ..prompts import report_planning_instructions, section_writing_instructions

load_dotenv()

# Two temperatures:
# - planner/writer uses 0.7 for creative, varied prose
# - assembler uses 0.0 — pure deterministic formatting, no creativity needed
llm        = ChatOpenAI(model="gpt-4o", temperature=0.7)
llm_strict = ChatOpenAI(model="gpt-4o", temperature=0.0)


class PlannedSection(BaseModel):
    """One section in the report outline produced by plan_report."""
    index:              int
    title:              str
    description:        str    = Field(..., description="What this section covers.")
    relevant_subtopics: List[str] = Field(..., description="Subtopic titles this section draws from.")
    visual_type:        Literal[
        "bar_chart", "line_chart", "pie_chart",
        "flow_diagram", "tree_diagram", "comparison_table", "none"
    ]
    visual_title:       str    = Field(default="", description="Title for the planned visual.")
    visual_description: str    = Field(default="", description="One sentence: what the visual shows.")


class ReportOutline(BaseModel):
    sections: List[PlannedSection] = Field(default_factory=list)

class SectionWritingState(BaseModel):
    """Private state for one parallel write_section branch."""
    topic:          str
    section:        PlannedSection
    relevant_notes: str
    written:        Optional[WrittenSection] = None


class WriterState(TypedDict):
    topic:            str
    researcher_notes: str
    outline:          List[PlannedSection]
    written_sections: Annotated[List[WrittenSection], operator.add]
    writer_draft:     str



def _slice_notes_for_section(
    researcher_notes: str,
    section: PlannedSection,
) -> str:
    """
    Extract the portion of researcher_notes relevant to this section by scanning
    for matching subtopic headings. Falls back to the full notes if none match.
    """
    lines = researcher_notes.split("\n")
    relevant, capture, found = [], False, False

    for line in lines:
        if line.startswith("## "):
            subtopic_name = line.split("—")[-1].strip() if "—" in line else ""
            capture = any(
                sub.lower() in subtopic_name.lower()
                for sub in section.relevant_subtopics
            )
            if capture:
                found = True
        if capture:
            relevant.append(line)

    return "\n".join(relevant) if found else researcher_notes


def _render_visual_html(visual: ReportVisual) -> str:
    """
    Convert a ReportVisual spec into a self-contained HTML snippet
    using Chart.js (charts) or inline SVG (diagrams/tables).
    Returns an empty string when visual_type is 'none'.
    """
    if visual.visual_type == "none":
        return ""

    vt   = visual.visual_type
    data = visual.data
    tid  = visual.title.replace(" ", "_").replace("/", "_")[:30]

    # ------------------------------------------------------------------ charts
    if vt in ("bar_chart", "line_chart", "pie_chart"):
        labels = json.dumps(data.get("labels", []))
        values = json.dumps(data.get("values", []))
        x_label = data.get("x_label", "")
        y_label = data.get("y_label", "")

        chart_type = {"bar_chart": "bar", "line_chart": "line", "pie_chart": "pie"}[vt]

        scales_cfg = ""
        if vt != "pie_chart":
            scales_cfg = f"""
              scales: {{
                x: {{ title: {{ display: true, text: '{x_label}' }} }},
                y: {{ title: {{ display: true, text: '{y_label}' }}, beginAtZero: true }}
              }},"""

        return f"""
<div style="margin:24px 0">
  <p style="font-weight:500;margin-bottom:8px">{visual.title}</p>
  <p style="font-size:0.85em;color:#666;margin-bottom:12px">{visual.description}</p>
  <div style="max-width:560px">
    <canvas id="chart_{tid}"></canvas>
  </div>
  <script src="https://cdn.jsdelivr.net/npm/chart.js@4/dist/chart.umd.min.js"></script>
  <script>
  (function(){{
    const ctx = document.getElementById('chart_{tid}');
    new Chart(ctx, {{
      type: '{chart_type}',
      data: {{
        labels: {labels},
        datasets: [{{
          label: '{visual.title}',
          data: {values},
          backgroundColor: [
            '#6366F1','#14B8A6','#F59E0B','#EF4444','#8B5CF6',
            '#06B6D4','#10B981','#F97316','#EC4899','#84CC16'
          ],
          borderRadius: 4,
          borderWidth: 1,
        }}]
      }},
      options: {{
        responsive: true,
        plugins: {{ legend: {{ display: {'true' if vt == 'pie_chart' else 'false'} }} }},{scales_cfg}
      }}
    }});
  }})();
  </script>
</div>"""

    # --------------------------------------------------------------- flow_diagram
    if vt == "flow_diagram":
        steps = data.get("steps", [])
        boxes = []
        arrows = []
        box_w, box_h, gap = 180, 44, 32
        total_h = len(steps) * (box_h + gap) + 40
        cx = 340

        for i, step in enumerate(steps):
            y = 20 + i * (box_h + gap)
            safe = step.replace("'", "&#39;")
            boxes.append(
                f'<rect x="{cx - box_w//2}" y="{y}" width="{box_w}" height="{box_h}" '
                f'rx="8" fill="#EEF2FF" stroke="#6366F1" stroke-width="1"/>'
                f'<text x="{cx}" y="{y + box_h//2}" text-anchor="middle" '
                f'dominant-baseline="central" font-size="13" fill="#312E81">{safe}</text>'
            )
            if i < len(steps) - 1:
                ay = y + box_h
                arrows.append(
                    f'<line x1="{cx}" y1="{ay}" x2="{cx}" y2="{ay + gap - 4}" '
                    f'stroke="#6366F1" stroke-width="1.5" '
                    f'marker-end="url(#arr_{tid})"/>'
                )

        nodes_svg = "\n".join(boxes + arrows)
        return f"""
<div style="margin:24px 0">
  <p style="font-weight:500;margin-bottom:8px">{visual.title}</p>
  <p style="font-size:0.85em;color:#666;margin-bottom:12px">{visual.description}</p>
  <svg width="100%" viewBox="0 0 680 {total_h}" xmlns="http://www.w3.org/2000/svg">
    <defs>
      <marker id="arr_{tid}" viewBox="0 0 10 10" refX="8" refY="5"
              markerWidth="6" markerHeight="6" orient="auto-start-reverse">
        <path d="M2 1L8 5L2 9" fill="none" stroke="#6366F1"
              stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round"/>
      </marker>
    </defs>
    {nodes_svg}
  </svg>
</div>"""

    # --------------------------------------------------------------- tree_diagram
    if vt == "tree_diagram":
        root     = data.get("root", "Root")
        children = data.get("children", [])
        node_w, node_h = 160, 40
        gap_x, gap_y   = 20, 60
        num_children   = max(len(children), 1)
        total_w        = num_children * (node_w + gap_x)
        total_h        = 180
        cx_root        = total_w // 2

        nodes_svg = (
            f'<rect x="{cx_root - node_w//2}" y="20" width="{node_w}" height="{node_h}" '
            f'rx="8" fill="#EEF2FF" stroke="#6366F1" stroke-width="1"/>'
            f'<text x="{cx_root}" y="40" text-anchor="middle" dominant-baseline="central" '
            f'font-size="13" font-weight="500" fill="#312E81">{root}</text>'
        )

        for i, child in enumerate(children):
            cx_child = (i + 0.5) * (node_w + gap_x)
            cy_child = 20 + node_h + gap_y
            label    = child.get("label", "").replace("'", "&#39;")
            nodes_svg += (
                f'<line x1="{cx_root}" y1="{20 + node_h}" x2="{cx_child}" y2="{cy_child}" '
                f'stroke="#6366F1" stroke-width="1" stroke-dasharray="4 2"/>'
                f'<rect x="{cx_child - node_w//2}" y="{cy_child}" width="{node_w}" height="{node_h}" '
                f'rx="6" fill="#F5F3FF" stroke="#8B5CF6" stroke-width="1"/>'
                f'<text x="{cx_child}" y="{cy_child + node_h//2}" text-anchor="middle" '
                f'dominant-baseline="central" font-size="12" fill="#4C1D95">{label}</text>'
            )

        return f"""
<div style="margin:24px 0">
  <p style="font-weight:500;margin-bottom:8px">{visual.title}</p>
  <p style="font-size:0.85em;color:#666;margin-bottom:12px">{visual.description}</p>
  <svg width="100%" viewBox="0 0 {total_w} {total_h}" xmlns="http://www.w3.org/2000/svg">
    {nodes_svg}
  </svg>
</div>"""

    # --------------------------------------------------------------- comparison_table
    if vt == "comparison_table":
        headers = data.get("headers", [])
        rows    = data.get("rows", [])

        th_cells = "".join(
            f'<th style="padding:10px 14px;background:#6366F1;color:#fff;'
            f'font-weight:500;text-align:left;white-space:nowrap">{h}</th>'
            for h in headers
        )
        row_html = ""
        for r_idx, row in enumerate(rows):
            bg = "#F5F3FF" if r_idx % 2 == 0 else "#fff"
            td_cells = "".join(
                f'<td style="padding:8px 14px;border-bottom:1px solid #E5E7EB;'
                f'font-size:0.9em">{cell}</td>'
                for cell in row
            )
            row_html += f'<tr style="background:{bg}">{td_cells}</tr>'

        return f"""
<div style="margin:24px 0;overflow-x:auto">
  <p style="font-weight:500;margin-bottom:8px">{visual.title}</p>
  <p style="font-size:0.85em;color:#666;margin-bottom:12px">{visual.description}</p>
  <table style="width:100%;border-collapse:collapse;border-radius:8px;overflow:hidden;
                box-shadow:0 1px 3px rgba(0,0,0,.1)">
    <thead><tr>{th_cells}</tr></thead>
    <tbody>{row_html}</tbody>
  </table>
</div>"""

    return ""   # fallback — unknown type


def _section_to_html(section: WrittenSection) -> str:
    """Convert a WrittenSection to a full HTML block."""
    import re
    # --- markdown → basic HTML ---
    prose = section.prose

    # ## heading
    prose = re.sub(r'^## (.+)$', r'<h2>\1</h2>', prose, flags=re.MULTILINE)
    # **bold**
    prose = re.sub(r'\*\*(.+?)\*\*', r'<strong>\1</strong>', prose)
    # bullet lists
    lines, in_list, out = prose.split("\n"), False, []
    for line in lines:
        stripped = line.lstrip()
        if stripped.startswith("- "):
            if not in_list:
                out.append("<ul>")
                in_list = True
            out.append(f"<li>{stripped[2:]}</li>")
        else:
            if in_list:
                out.append("</ul>")
                in_list = False
            if line.strip():
                out.append(f"<p>{line}</p>" if not line.startswith("<") else line)
    if in_list:
        out.append("</ul>")
    prose_html = "\n".join(out)

    # key points
    kp_items = "".join(f"<li>{kp}</li>" for kp in section.key_points)
    key_points_html = (
        f'<div style="background:#F0FDF4;border-left:3px solid #22C55E;'
        f'border-radius:0 8px 8px 0;padding:12px 16px;margin:16px 0">'
        f'<p style="font-weight:500;margin:0 0 8px">Key takeaways</p>'
        f'<ul style="margin:0;padding-left:18px">{kp_items}</ul></div>'
        if kp_items else ""
    )

    visual_html = _render_visual_html(section.visual)

    return f"""
<section style="margin-bottom:40px">
  {prose_html}
  {visual_html}
  {key_points_html}
</section>"""

# ---------------------------------------------------------------------------
def plan_report(state: WriterState) -> dict:
    """
    Read the full researcher_notes + topic breakdown and produce a
    structured section-by-section outline for the report.
    """
    structured_llm = llm_strict.with_structured_output(ReportOutline)
    notes_for_plan = state["researcher_notes"][:15_000]

    outline = structured_llm.invoke([
        SystemMessage(content=report_planning_instructions.format(
            topic=state["topic"]
        )),
        HumanMessage(content=(
            f"Research notes:\n\n{notes_for_plan}\n\n"
            "Produce a structured outline for a comprehensive report on this topic."
        )),
    ])

    return {"outline": outline.sections}


def dispatch_sections(state: WriterState) -> List[Send]:
    """
    Fan-out: one Send per planned section.
    Each branch gets its own SectionWritingState with only the notes it needs.
    """
    return [
        Send(
            "write_section",
            SectionWritingState(
                topic          = state["topic"],
                section        = section,
                relevant_notes = _slice_notes_for_section(
                    state["researcher_notes"], section
                ),
            ),
        )
        for section in state["outline"]
    ]


def write_section(state: SectionWritingState) -> dict:
    structured_llm = llm.with_structured_output(WrittenSection, method="function_calling")

    written = structured_llm.invoke([
        SystemMessage(content=section_writing_instructions.format(
            topic               = state.topic,
            section_index       = state.section.index,
            section_title       = state.section.title,
            relevant_notes      = state.relevant_notes,
            visual_type         = state.section.visual_type,
            visual_title        = state.section.visual_title,
            visual_description  = state.section.visual_description,
        )),
        HumanMessage(content=(
            f"Write section {state.section.index}: '{state.section.title}'. "
            f"Description: {state.section.description}"
        )),
    ])

    return {"written_sections": [written]}


def assemble_report(state: WriterState) -> dict:
    """
    Collect all WrittenSection objects (already merged by operator.add),
    sort by index, render each to HTML, and build the final report string.
    """
    sections = sorted(state["written_sections"], key=lambda s: s.index)
    topic    = state["topic"]

    html_parts = [
        f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>{topic}</title>
  <style>
    body      {{ font-family: system-ui, sans-serif; max-width: 860px; margin: 0 auto;
                padding: 40px 24px; color: #111; line-height: 1.7; }}
    h1        {{ font-size: 2em; font-weight: 600; border-bottom: 2px solid #6366F1;
                padding-bottom: 12px; color: #1E1B4B; }}
    h2        {{ font-size: 1.35em; font-weight: 500; color: #312E81; margin-top: 36px; }}
    p         {{ margin: 0 0 14px; }}
    ul        {{ margin: 0 0 14px; padding-left: 20px; }}
    li        {{ margin-bottom: 6px; }}
    strong    {{ color: #1E1B4B; }}
    section   {{ border-top: 1px solid #E5E7EB; padding-top: 28px; }}
  </style>
</head>
<body>
  <h1>{topic}</h1>
  <p style="color:#6B7280;font-size:0.9em">
    Generated by the Research Agent · {len(sections)} sections
  </p>
"""
    ]

    for section in sections:
        html_parts.append(_section_to_html(section))

    html_parts.append("</body>\n</html>")
    report_html = "\n".join(html_parts)

    return {"writer_draft": report_html}


writer_builder = StateGraph(WriterState)
writer_builder.add_node("plan_report",      plan_report)
writer_builder.add_node("write_section",    write_section)
writer_builder.add_node("assemble_report",  assemble_report)

writer_builder.add_edge(START, "plan_report")

# plan_report → dispatch_sections (conditional edge returning Send objects)
writer_builder.add_conditional_edges(
    "plan_report",
    dispatch_sections,
)

# Every parallel write_section branch converges on assemble_report
writer_builder.add_edge("write_section",   "assemble_report")
writer_builder.add_edge("assemble_report", END)

writer_graph = writer_builder.compile()


def run_writer(breakdown: TopicBreakdownResult, researcher_notes: str) -> str:
    """
    Produces the final HTML report.

    Usage in supervisor:
        draft = run_writer(overall['topic_breakdown'], overall['researcher_notes'])
        overall['writer_draft'] = draft
    """
    initial_state: WriterState = {
        "topic":            breakdown.Topic,
        "researcher_notes": researcher_notes,
        "outline":          [],
        "written_sections": [],
    }

    final = {}
    for event in writer_graph.stream(initial_state, stream_mode="values"):
        final = event

    return final.get("writer_draft", "")


def writer_node(state: overallState) -> dict:
    """LangGraph node wrapper: runs the writer sub-graph and writes the
    finished HTML report string back into overallState."""
    draft = run_writer(state["topic_breakdown"], state["researcher_notes"])
    return {
        "writer_draft": draft,
        "messages": [
            AIMessage(
                content="Report written. Final HTML report is ready.",
                name="Writer",
            )
        ],
    }