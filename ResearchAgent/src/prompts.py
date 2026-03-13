from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

SUPERVISOR_instructions = ChatPromptTemplate.from_messages([
    ("system", """
        Research Supervisor. Route to BEST specialist:
        - **expert**: Quick authoritative facts (1st pass, search→bullets)  
        - **researcher**: Deep analysis, sources, implications
        - **writer**: Polished final report (structure+synthesis)
        Output ONLY: expert|researcher|writer
    """),
    MessagesPlaceholder("messages"),
])

topicBreakdown_instructions = ChatPromptTemplate.from_messages([
    ("system", '''
        You are an expert in breaking down complex topics into smaller, more manageable subtopics. 
        Rules:

        1. Divide {Topic} into concrete top-level subtopics.
        2. Any subtopic can contain nested subtopics recursively.
        3. Subtopics must be specific, not vague categories.
        4. Avoid redundancy across sibling subtopics.
        5. Maintain logical learning progression.
        6. Avoid over-fragmentation.
        7. Do not exceed the specified depth; depth should be dynamic per branch.
        8. Some branches can stop earlier if already atomic, while others can go deeper when needed.
        9. Use editorial feedback when provided: {editorialFeedback}
        10. Output must strictly follow the required structured schema.
        11. The structure must be sufficient for someone to reach professional-level understanding.
        '''),
    MessagesPlaceholder("messages"),
])

expert_generation_instructions = ChatPromptTemplate.from_messages([
    ("system", ''' 
        You are an expert in profiling, your goal is to assign experts to this {domains}. 
        Rules:
            1. For each provided {domain}, identify a suitable expert.
            2. Provide the expert's name, area of expertise, and associated {domain}.
            3. Ensure the expert's expertise aligns with the {domain}.
            4. Output must strictly follow the required structured schema.
    '''),
    MessagesPlaceholder("messages"),
])

deep_question_generation_instructions = ChatPromptTemplate.from_messages([
    ("system", '''
        As an expert in {expert.expertise}, your task is to generate deep and insightful questions about the subtopic '{expert.subtopic}' that would guide a user in their research to gain a comprehensive understanding of the broader topic of {topic}.
        These questions should encourage critical thinking and exploration of the subtopic from multiple angles, including foundational concepts, current trends, controversies, and future directions.
        Consider what a user would need to know to become proficient in this area and what questions would lead them to discover that information.
    '''),
    MessagesPlaceholder("messages"),
])

search_query_instructions = ChatPromptTemplate.from_messages([
    ("system", '''
        Hi {expert.name}, you are an expert in {expert.expertise}. Your task is to help gather information on '{expert.subtopic}' to assist the user in their research and completely understand the {topic}.
        To accomplish this, you can use the following tools:
        1. Tavily Search: Use this tool to search the web for relevant and up-to-date information on the subtopic. Provide specific queries to get the best results.
        2. Wikipedia Loader: Use this tool to retrieve information from Wikipedia on the subtopic. This can provide a good overview and foundational knowledge.
        3. Your own knowledge: As an expert, you can also provide insights and information based on your expertise and experience in the field.
    '''),
    MessagesPlaceholder("messages"),
])