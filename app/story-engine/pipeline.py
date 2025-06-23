from crewai import Task
from rag import store_narrative, retrieve_context
from agents import get_agents

def story_turn(genre, user_input):
    context = retrieve_context(genre, user_input)
    narrator, overseer, image_prompter, next_prompt_agent = get_agents(genre)

    narration_task = Task(
        description=f"""CONTEXT:\n{context}\n\nUSER INPUT:\n{user_input}\n\nWrite the next story scene.""",
        expected_output="Immersive story scene",
        agent=narrator
    )
    story_text = narration_task.run()

    overseer_task = Task(
        description=f"""
Review the following story:
\"\"\"{story_text}\"\"\"

Check for consistency with world, character behavior, and tone.
If consistent, say 'Approved'. Else, describe the issues clearly.
""",
        expected_output="Approval or detailed feedback",
        agent=overseer
    )
    feedback = overseer_task.run()

    if "approved" not in feedback.lower():
        revise_task = Task(
            description=f"""
Revise the following story using this feedback:
FEEDBACK: {feedback}

STORY:
{story_text}
""",
            expected_output="Revised story scene",
            agent=narrator
        )
        story_text = revise_task.run()

    store_narrative(genre, story_text)

    image_task = Task(
        description=f"Convert this scene into a one-line image generation prompt: {story_text}",
        expected_output="One-line visual description",
        agent=image_prompter
    )
    image_prompt = image_task.run()

    next_prompt_task = Task(
        description=f"Suggest 2-3 next user actions based on this: {story_text}",
        expected_output="Decision list",
        agent=next_prompt_agent
    )
    next_prompt = next_prompt_task.run()

    return {
        "story": story_text,
        "feedback": feedback,
        "image_prompt": image_prompt,
        "next_prompt": next_prompt
    }
