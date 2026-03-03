from agent.graph import graph


def print_subtopic(subtopic, experts_map, indent=0):
    """Recursively print subtopic tree with associated experts."""
    prefix = "  " * indent
    print(f"{prefix}!! {subtopic.title}")
    print(f"{prefix}   {subtopic.description}")
    
    # Print experts for this subtopic
    if subtopic.title in experts_map:
        for expert in experts_map[subtopic.title]:
            print(f"{prefix}   @ Expert: {expert.name} ({expert.expertise})")
    
    # Recursively print nested subtopics
    for child in subtopic.subtopics:
        print_subtopic(child, experts_map, indent + 1)


def print_results(topic, result):
    """Pretty print the research breakdown with experts."""
    print("\n" + "=" * 60)
    print(f" TOPIC: {topic}")
    print("=" * 60 + "\n")
    
    # Build experts lookup by subtopic
    experts_map = {}
    for expert in result.get('experts', []):
        if expert.subtopic not in experts_map:
            experts_map[expert.subtopic] = []
        experts_map[expert.subtopic].append(expert)
    
    # Print each top-level subtopic tree
    for subtopic in result.get('subTopics', []):
        print_subtopic(subtopic, experts_map)
        print()


def main():
    while True:
        input_topic = input("Enter a topic you'd like to explore (or 'exit' to quit): ")
        
        if input_topic.lower() == 'exit':
            break

        state = {
            'Topic': input_topic,
            'estimatedDepth': 3,
        }
        result = graph.invoke(state)
        print_results(input_topic, result)


if __name__ == "__main__":
    main()