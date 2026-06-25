import json
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Load FAQ data
with open("faq_data.json", "r") as file:
    data = json.load(file)

questions = [item["question"] for item in data["questions"]]
answers = [item["answer"] for item in data["questions"]]

# Convert questions into vectors
vectorizer = TfidfVectorizer()
question_vectors = vectorizer.fit_transform(questions)

print("FAQ Chatbot Started")
print("Type 'exit' to quit.\n")

while True:
    user_input = input("You: ")

    if user_input.lower() == "exit":
        print("Chatbot: Goodbye!")
        break

    user_vector = vectorizer.transform([user_input])

    similarity_scores = cosine_similarity(
        user_vector,
        question_vectors
    )

    best_match_index = similarity_scores.argmax()

    print("Chatbot:", answers[best_match_index])