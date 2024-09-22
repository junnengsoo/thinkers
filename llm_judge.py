import weave
import os
from openai import OpenAI
from dotenv import load_dotenv
import json

load_dotenv()
os.environ['OPENROUTER_API_KEY'] = os.getenv('OPENROUTER_API_KEY')

client = OpenAI(
    api_key=os.environ["OPENROUTER_API_KEY"],
    base_url="https://openrouter.ai/api/v1",
)

@weave.op()
def evaluate_pairwise(task: str, ground_truth: str, solutions: list) -> dict:
    evaluations = []
    
    # Evaluate each pair of solutions
    for i in range(len(solutions)):
        for j in range(i + 1, len(solutions)):
            reasoning_1, solution_1 = solutions[i]
            reasoning_2, solution_2 = solutions[j]
            print(solutions)
            
            response = client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {
                        "role": "system",
                        "content": """You are an expert judge evaluating two solution processes for the same word problem. 
                        Compare the reasoning processes and final solutions, assessing their coherence, relevance, and correctness. 
                        Provide your evaluation in JSON format with the following structure:
                        {
                            "winner": "way_1" or "way_2", // Indicate which way is preferred
                            "reason": string // Brief explanation for the preference
                        }"""
                    },
                    {
                        "role": "user",
                        "content": (
                            f"Word Problem: {task}\nCorrect Answer: {ground_truth}\n\n"
                            f"Way 1 - Reasoning: {reasoning_1}\nSolution: {solution_1}\n\n"
                            f"Way 2 - Reasoning: {reasoning_2}\nSolution: {solution_2}"
                        )
                    }
                ],
                response_format={"type": "json_object"}
            )
            comparison_result = response.choices[0].message.content
            
            # Parse the JSON string into a dictionary
            evaluations.append(json.loads(comparison_result))  # Parse here
    
    # Aggregate results into a ranking
    ranking = {}
    for result in evaluations:
        winner = result["winner"]
        if winner not in ranking:
            ranking[winner] = 0
        ranking[winner] += 1
    
    ranked_results = sorted(ranking.items(), key=lambda x: x[1], reverse=True)
    
    return {
        "pairwise_evaluations": evaluations,
        "final_ranking": ranked_results
    }


# Example usage
def main():
    task = "A train travels 60 miles in 1 hour. How far will it travel in 2 hours?"
    solutions = [
        ("The train travels at 60 miles per hour. In 2 hours, it will travel 60 * 2 = 120 miles.", "120 miles"),
        ("Since the train is consistent at 60 miles per hour, in 2 hours, it will cover double that distance: 120 miles.", "120 miles"),
        ("In 2 hours, at 60 miles per hour, the distance traveled would be calculated as speed multiplied by time: 60 * 2 = 120 miles.", "120 miles")
    ]
    
    results = evaluate_pairwise(task, solutions)    
    # Print pairwise evaluations
    print("Pairwise Evaluations:")
    for index, evaluation in enumerate(results["pairwise_evaluations"]):
        way = index + 1  # Way 1, Way 2, etc.
        reasoning, solution = solutions[index]
        print(f"CoT Way {way}: {reasoning}")
        print(f"Way {way} Result: {solution}")
    
    # Print the winner and evaluation reasoning
    for evaluation in results["pairwise_evaluations"]:
        if evaluation["winner"] == "way_1":
            winning_way = "Way 1"
        else:
            winning_way = "Way 2"
        
        print(f"\nWinner: {winning_way}")
        print(f"Evaluation Reasoning: {evaluation['reason']}")

# Run the main function
if __name__ == "__main__":
    main()
