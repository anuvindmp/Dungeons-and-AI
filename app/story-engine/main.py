from pipeline import story_turn
from rag import seed_genre_documents
import os

def main():
    genres = ["horror", "medieval", "scifi"]
    print("🎮 Welcome to Dungeons & AI")
    print("Select a genre:")
    for i, g in enumerate(genres):
        print(f"{i + 1}. {g.capitalize()}")
    genre = genres[int(input("Enter choice (1-3): ")) - 1]

    # Seed lore if first time
    if not os.path.exists(f"./rag_store/{genre}"):
        seed_genre_documents(genre, f"./lore/{genre}")

    while True:
        user_input = input("\n🗨️  Your action: ")
        result = story_turn(genre, user_input)

        print("\n📖 STORY:\n", result["story"])
        print("\n🧠 FEEDBACK:\n", result["feedback"])
        print("\n🎨 IMAGE PROMPT:\n", result["image_prompt"])
        print("\n🎲 NEXT PROMPTS:\n", result["next_prompt"])

if __name__ == "__main__":
    main()
