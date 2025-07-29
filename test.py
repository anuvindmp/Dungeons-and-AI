import os
import json
import re
import ast
import sys
import pprint
import nltk
nltk.download('punkt_tab')
from crewai import Agent, Task, Crew, Process
from langchain_groq import ChatGroq
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import Chroma
from transformers import AutoModelForCausalLM, AutoTokenizer
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from rouge_score import rouge_scorer

nltk.download('punkt')

# === Environment Setup ===
os.environ["GROQ_API_KEY"] = "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"

embedding_fn = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

vector_store = Chroma(
    collection_name="story_memory",
    embedding_function=embedding_fn,
    persist_directory="./rag_db"
)

llm = ChatGroq(
    groq_api_key=os.environ["GROQ_API_KEY"],
    model_name="groq/llama-3.3-70b-versatile",
    temperature=0.8
)

# === Agents ===
story_agent = Agent(
    role="Storyteller",
    goal=(
        "Craft a vivid, immersive story segment based on the user's input and past context. "
        "At the end of the story, provide four creative options for how the story could continue. "
        "These options must feel natural and build on the current events."
    ),
    backstory=(
        "Once a wandering bard who collected myths from forgotten realms, the Storyteller now brings worlds to life through narrative."
    ),
    allow_delegation=False,
    llm=llm
)

lore_judge_agent = Agent(
    role="Lore Keeper",
    goal="Evaluate each story for alignment with established lore and narrative consistency. Score and highlight any discrepancies.",
    backstory="A grand librarian of the Eternal Archives.",
    allow_delegation=False,
    llm=llm
)

suggestion_agent = Agent(
    role="Narrative Editor",
    goal="Offer suggestions to improve story fragments that fail to align with the world’s lore or quality expectations.",
    backstory="A brilliant novelist turned editor.",
    allow_delegation=False,
    llm=llm
)

memory_agent = Agent(
    role="Story Archivist",
    goal="Summarize accepted story pieces and store them in memory for future retrieval and world continuity.",
    backstory="An ancient AI record-keeper.",
    allow_delegation=False,
    llm=llm
)

# === Tasks ===
generate_story_task = Task(
    description="Generate immersive fantasy story with 4 continuation options.",
    expected_output="JSON: {story: <str>, options: [<str>, <str>, <str>, <str>]}",
    agent=story_agent
)

evaluate_lore_task = Task(
    description="Evaluate story for lore consistency and score from 0-10.",
    expected_output="Dict: {score: int, issues_found: list}",
    agent=lore_judge_agent
)

suggest_fix_task = Task(
    description="Given flaws, suggest constructive improvements.",
    expected_output="List of suggestions.",
    agent=suggestion_agent
)

summarize_task = Task(
    description="Summarize accepted story into 2-3 lore-relevant lines.",
    expected_output="Concise story summary.",
    agent=memory_agent
)

crew = Crew(
    agents=[story_agent, lore_judge_agent, suggestion_agent, memory_agent],
    tasks=[generate_story_task, evaluate_lore_task, suggest_fix_task, summarize_task],
    verbose=True
)

# === Parsing Utilities ===
def parse_story_output(raw_output):
    try:
        story_data = ast.literal_eval(raw_output)
        return story_data["story"], story_data["options"]
    except Exception as e:
        print("Could not parse story output:", e, file=sys.stderr)
        return raw_output, []

def parse_score_output(raw_output):
    try:
        if isinstance(raw_output, dict):
            return raw_output["score"], raw_output.get("issues_found", [])
        score_data = json.loads(raw_output.strip())
        return score_data["score"], score_data.get("issues_found", [])
    except Exception as e:
        print("Could not parse score output:", e, file=sys.stderr)
        return 0, ["Failed to parse score"]

def parse_nested_story_output(story_blob):
    if isinstance(story_blob, str):
        try:
            return json.loads(story_blob)
        except json.JSONDecodeError:
            pass
        try:
            story_match = re.search(r'"story"\s*:\s*"(.+?)"\s*,\s*"options"', story_blob, re.DOTALL)
            story_text = story_match.group(1).strip() if story_match else "Could not extract story"
            options_match = re.findall(r'\d+\.\s*"(.*?)"', story_blob, re.DOTALL)
            return {"story": story_text, "options": [opt.strip() for opt in options_match]}
        except Exception as e:
            print(f"Failed to repair story output: {e}", file=sys.stderr)
            return {"story": story_blob, "options": []}
    return story_blob

# === Main Story Generation Function ===
def run_story_loop(user_prompt, vector_store, threshold=7, max_attempts=3):
    attempts = 0
    suggestions = None
    story_output = None
    options = None

    while attempts < max_attempts:
        print(f"\nAttempt #{attempts + 1}", file=sys.stderr)

        rag_context = ""
        try:
            rag_docs = vector_store.similarity_search(user_prompt, k=2)
            rag_context = "\n".join([doc.page_content for doc in rag_docs])
        except Exception as e:
            print("Could not retrieve RAG context:", e, file=sys.stderr)

        story_output = story_agent.execute_task(
            generate_story_task,
            context={
                "user_prompt": user_prompt,
                "rag_context": rag_context,
                "suggestions": suggestions or ""
            }
        )

        story_text, options = parse_story_output(story_output)

        score_output = lore_judge_agent.execute_task(
            evaluate_lore_task,
            context={"generated_story": story_text, "rag_context": rag_context}
        )
        score, issues = parse_score_output(score_output)
        print(f"\n Score: {score}/10", file=sys.stderr)

        if score >= threshold:
            summary = memory_agent.execute_task(
                summarize_task,
                context={"accepted_story": story_text}
            )
            try:
                vector_store.add_texts([summary])
                vector_store.persist()
                print("Summary stored in RAG DB.", file=sys.stderr)
            except Exception as e:
                print("Could not store in RAG DB:", e, file=sys.stderr)
            break
        else:
            print("Story needs improvement.", file=sys.stderr)
            suggestions_output = suggestion_agent.execute_task(
                suggest_fix_task,
                context={"generated_story": story_text, "issues_found": issues}
            )
            suggestions = suggestions_output
            attempts += 1

    return {"story": story_text, "options": options}

# === Evaluation Mode ===
def evaluate_story_model():
    prompts = [
        "A knight enters a cursed forest.",
        "A thief sneaks into a royal vault.",
        "A wizard awakens a forgotten golem."
    ]

    reference_outputs = [
        "The knight bravely stepped into the eerie, fog-covered forest, unsure of the dangers that lay ahead.",
        "With silent steps, the thief crept through the vault shadows, eyes fixed on the shimmering crown.",
        "The ancient golem stirred to life as the wizard's incantation echoed across the ruins."
    ]

    generated_outputs = []

    for prompt in prompts:
        result = run_story_loop(prompt, vector_store)
        parsed = parse_nested_story_output(result.get("story", ""))
        generated_outputs.append(parsed.get("story", "").strip())

    # Compute BLEU and ROUGE-L
    smoothie = SmoothingFunction().method4
    scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)

    bleu_scores = []
    rouge_l_scores = []

    for hypo, ref in zip(generated_outputs, reference_outputs):
        hypo_tokens = nltk.word_tokenize(hypo)
        ref_tokens = [nltk.word_tokenize(ref)]
        bleu = sentence_bleu(ref_tokens, hypo_tokens, smoothing_function=smoothie)
        rouge_l = scorer.score(ref, hypo)['rougeL'].fmeasure
        bleu_scores.append(bleu)
        rouge_l_scores.append(rouge_l)

    print(f"\n--- Evaluation Metrics ---")
    print(f"Average BLEU Score: {sum(bleu_scores)/len(bleu_scores):.4f}")
    print(f"Average ROUGE-L Score: {sum(rouge_l_scores)/len(rouge_l_scores):.4f}")

# === Main Entry Point ===
if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1].lower() == "eval":
        evaluate_story_model()
    else:
        prompt = input("Enter your initial prompt to start the story: ")
        result = run_story_loop(prompt.strip('"'), vector_store)
        story_blob = result.get("story", "")
        story_parsed = parse_nested_story_output(story_blob)

        final_output = {
            "story": story_parsed.get("story", "").strip(),
            "options": [opt.strip() for opt in story_parsed.get("options", [])]
        }

        print(json.dumps(final_output, indent=2))
