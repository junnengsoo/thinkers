import os
from openai import OpenAI
import weave
from dotenv import load_dotenv
from llm_judge import evaluate_word_problem_reasoning  # Import our custom function from llm_judge.py

load_dotenv()

weave.init('together-weave')

os.environ['OPENROUTER_API_KEY'] = os.getenv('OPENROUTER_API_KEY')

client = OpenAI(
    api_key=os.environ["OPENROUTER_API_KEY"],
    base_url="https://openrouter.ai/api/v1",
)

def generate_word_problem_and_solution():
    system_content = "You are a math teacher creating word problems. Provide a word problem, its step-by-step solution, and the final answer."
    user_content = "Create a word problem about travel or transportation."

    chat_completion = client.chat.completions.create(
        model="openai/gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": system_content},
            {"role": "user", "content": user_content},
        ],
        temperature=0.7,
        max_tokens=1024,
    )
    response = chat_completion.choices[0].message.content
    return response

def parse_problem_and_solution(response):
    sections = response.split("\n\n")
    problem = sections[0].strip()
    solution = sections[-1].strip()
    reasoning = "\n".join(sections[1:-1]).strip()
    return problem, reasoning, solution

# Generate word problem and solution
generated_content = generate_word_problem_and_solution()
print("Generated content:\n", generated_content)

# Parse the generated content
problem, reasoning, solution = parse_problem_and_solution(generated_content)

# Evaluate the word problem solution using the function from llm_judge.py
evaluation = evaluate_word_problem_reasoning(problem, reasoning, solution)
print("\nEvaluation results:")
print(evaluation)

# Optional: Analyze the evaluation
evaluation_dict = eval(evaluation)  # Convert string to dictionary
overall_score = evaluation_dict['overall_score']
feedback = evaluation_dict['feedback']

print(f"\nOverall score: {overall_score}")
print(f"Feedback: {feedback}")