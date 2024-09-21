import weave
import os
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()
os.environ['OPENROUTER_API_KEY'] = os.getenv('OPENROUTER_API_KEY')

client = OpenAI(
    api_key=os.environ["OPENROUTER_API_KEY"],
    base_url="https://openrouter.ai/api/v1",
)

@weave.op()
def evaluate_word_problem_reasoning(problem: str, reasoning: str, solution: str) -> dict:
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {
                "role": "system",
                "content": """You are an expert judge evaluating the solution process for word problems. 
                Given a word problem, the reasoning process, and the final solution, assess their coherence, relevance, and correctness. 
                Provide your evaluation in JSON format with the following structure:
                {
                    "problem_comprehension": float, // 0-1 score on how well the reasoning demonstrates understanding of the problem
                    "solution_strategy": float, // 0-1 score on the appropriateness and effectiveness of the problem-solving approach
                    "calculation_accuracy": float, // 0-1 score on the correctness of any calculations or logical deductions
                    "answer_correctness": float, // 0-1 score on how well the final answer addresses the problem question
                    "overall_score": float, // 0-1 score summarizing the overall quality of the solution process
                    "feedback": string // Brief explanation of the evaluation and suggestions for improvement
                }"""
            },
            {
                "role": "user",
                "content": f"Word Problem: {problem}\n\nReasoning: {reasoning}\n\nSolution: {solution}"
            }
        ],
        response_format={ "type": "json_object" }
    )
    return response.choices[0].message.content