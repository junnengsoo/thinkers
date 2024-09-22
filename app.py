from datasets import load_dataset
from mcts import OpenAIModel, MathMCST
import weave
import json

weave.init('together-weave')

# Dataset Loading Function
def load_math_dataset(num_samples: int = 5):
    """
    Load the MATH dataset and extract the first `num_samples` questions.
    """
    print(f"Loading MATH dataset with {num_samples} samples...")
    ds = load_dataset("lighteval/MATH", "all")
    
    questions = []
    for i in range(num_samples):
        questions.append(ds['train'][i]['problem'])

    return questions

# Main app function
@weave.op()
def main():
    # Load the dataset
    math_questions = load_math_dataset(num_samples=5)

    # Initialize OpenAI Model (with Weave integration)
    model = OpenAIModel()

    # Process each question using MCTS and LLM judge
    for i, question in enumerate(math_questions):
        print(f"\nProcessing question {i+1}: {question}")
        # Create an instance of the MCTS solver with the question
        mcst = MathMCST(initial_task=question, model=model)

        # Solve the task using the MCTS and LLM judging mechanism
        dpo_results = mcst.solve_task()

        # Output the results
        print(f"\nDPO Results for question {i+1}:")
        for result in dpo_results:
            print(result)  # You can format this output as needed

if __name__ == "__main__":
    main()
