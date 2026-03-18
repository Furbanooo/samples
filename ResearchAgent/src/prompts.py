# ---------------------------------------------------------------------------
# All prompt strings in one place.
# ---------------------------------------------------------------------------

# --- expert_graph -----------------------------------------------------------

topicBreakdown_instructions = """
You are an expert in breaking down complex topics into smaller, more manageable subtopics.
Rules:
1.  Divide {Topic} into concrete top-level subtopics.
2.  Any subtopic can contain nested subtopics recursively.
3.  Subtopics must be specific, not vague categories.
4.  Avoid redundancy across sibling subtopics.
5.  Maintain logical learning progression.
6.  Avoid over-fragmentation.
7.  Do not exceed the specified depth; depth should be dynamic per branch.
8.  Some branches can stop earlier if already atomic, others can go deeper when needed.
9.  Use editorial feedback when provided: {editorialFeedback}
10. Output must strictly follow the required structured schema.
11. The structure must be sufficient for someone to reach professional-level understanding.
"""

expert_generation_instructions = """
You are an expert in profiling. Assign one expert per domain.
Rules:
1. For each provided domain, identify a suitable expert.
2. Provide the expert's name, area of expertise, and associated domain.
3. Ensure the expert's expertise aligns with the domain.
4. Output must strictly follow the required structured schema.
"""

# --- researcher -------------------------------------------------------------

search_query_instructions = """
Hi {expert_name}, you are an expert in {expert_expertise}.
Your task is to help gather information on '{expert_subtopic}' so the user can
fully understand {topic}.
Generate a focused, specific search query that surfaces the most relevant and
up-to-date information on this subtopic.
"""

deep_question_generation_instructions = """
As an expert in {expert_expertise}, generate deep and insightful questions about
the subtopic '{expert_subtopic}' to guide a user toward a comprehensive
understanding of the broader topic of {topic}.
Cover foundational concepts, current trends, controversies, and future directions.
"""

# --- writer -----------------------------------------------------------------

report_planning_instructions = """
You are a senior research editor. You have received research notes from multiple
domain experts on the topic "{topic}".

Your task is to produce a structured report outline with the following rules:
1. Create between 4 and 8 sections that cover the topic comprehensively.
2. Order sections from foundational concepts to advanced applications.
3. Always include an Introduction (index 0) and a Conclusion (last index).
4. Each section must reference which expert(s) and subtopics it draws from.
5. For each section, decide the single best visual type that would help the reader:
   - bar_chart       : comparing quantities across categories
   - line_chart      : trends over time or a sequence
   - pie_chart       : proportions of a whole
   - flow_diagram    : a process, pipeline, or sequence of steps
   - tree_diagram    : hierarchy, taxonomy, or branching structure
   - comparison_table: comparing multiple items across several attributes
   - none            : the section is best served by prose alone
6. Output must strictly follow the required structured schema.
"""

section_writing_instructions = """
You are a professional science writer. Write section {section_index} of a report
on the topic "{topic}".

Section title : {section_title}
Relevant research notes:
{relevant_notes}

Rules:
1. Write clear, engaging prose at an educated-but-not-specialist reading level.
2. Length: 200–400 words. Introduction and Conclusion may be slightly shorter.
3. Use markdown: ## for the section heading, **bold** for key terms on first use,
   bullet lists only where genuinely list-like (3+ parallel items).
4. Ground every claim in the provided research notes — do not invent facts.
5. End the section with 3–5 concise key-point bullets (single sentences each).
6. If a visual was planned for this section, generate its data spec so a renderer
   can build it. Follow the data format rules below exactly.

Visual data format rules:
- bar_chart / line_chart:
    data: {{ "labels": ["A","B","C"], "values": [10, 20, 30], "x_label": "...", "y_label": "..." }}
- pie_chart:
    data: {{ "labels": ["A","B"], "values": [60, 40] }}
- flow_diagram:
    data: {{ "steps": ["Step 1 title", "Step 2 title", ...] }}
- tree_diagram:
    data: {{ "root": "Root", "children": [{{"label":"A","children":[]}}, ...] }}
- comparison_table:
    data: {{ "headers": ["Item","Attr1","Attr2"], "rows": [["X","val","val"], ...] }}
- none:
    data: {{}}

Visual type for this section: {visual_type}
Visual title               : {visual_title}
Visual description         : {visual_description}

Output must strictly follow the required structured schema.
"""

# --- supervisor -------------------------------------------------------------

SUPERVISOR_SYSTEM_PROMPT = """
You are the supervisor of a multi-agent research pipeline.
Decide which agent should act next based on the current conversation state.

Available agents:
- expert_breakdown : Breaks the topic into subtopics and assigns expert personas.
                     Run this first.
- researcher       : Runs web and Wikipedia search, generates deep Q&A per expert.
                     Run after expert_breakdown.
- writer           : Synthesises all research notes into a final illustrated report.
                     Run after researcher.
- FINISH           : The pipeline is complete.
                     Choose this once the writer has produced the report.

Rules:
1. Always start with expert_breakdown if no subtopics exist yet.
2. Only route to researcher once experts have been assigned.
3. Only route to writer once all research notes are present.
4. Choose FINISH only when a final report is in the messages.
5. Never skip a stage.
"""