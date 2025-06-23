from crewai import Agent
from crewai.tools.base import BaseTool
from crewai.llms import BaseLLM
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from rag import retrieve_context

class FineTunedNarrativeLLM(BaseLLM):
    def __init__(self, model_path="./model"):
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModelForCausalLM.from_pretrained(model_path)
        self.pipeline = pipeline("text-generation", model=self.model, tokenizer=self.tokenizer)

    def complete(self, prompt: str, **kwargs) -> str:
        return self.pipeline(prompt, max_new_tokens=400, temperature=0.9, do_sample=True)[0]["generated_text"]

class RAGRetriever(BaseTool):
    def __init__(self, genre):
        self.genre = genre
        self.name = "RAG Retriever"
        self.description = f"Retrieves story-relevant context from the {genre} lore and memory."

    def _run(self, input: str) -> str:
        return retrieve_context(self.genre, input)

def get_agents(genre):
    retriever_tool = RAGRetriever(genre)

    narrator = Agent(
        role="Narrator",
        goal="Generate immersive narrative",
        backstory=f"A master storyteller in the {genre} genre.",
        llm=FineTunedNarrativeLLM(),
        tools=[retriever_tool]
    )

    overseer = Agent(
        role="Overseer",
        goal="Ensure story consistency and tone",
        backstory="A strict critic with deep knowledge of lore.",
        llm="gpt-4"
    )

    image_prompter = Agent(
        role="Image Prompter",
        goal="Create image prompts from story scenes",
        backstory="A visual artist AI.",
        llm="gpt-4"
    )

    next_prompt_agent = Agent(
        role="Prompt Suggester",
        goal="Suggest next user decisions",
        backstory="A branching narrative designer.",
        llm="gpt-4"
    )

    return narrator, overseer, image_prompter, next_prompt_agent
