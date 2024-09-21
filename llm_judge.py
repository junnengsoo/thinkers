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

# Example usage
word_problem = """
A train leaves New York at 2:00 PM traveling west at 60 miles per hour. 
Another train leaves Chicago at 3:00 PM traveling east at 70 miles per hour. 
If the distance between New York and Chicago is 800 miles, at what time will the two trains meet?
"""

reasoning = """
1. Understand the given information:
   - Train 1: Leaves New York at 2:00 PM, travels west at 60 mph
   - Train 2: Leaves Chicago at 3:00 PM, travels east at 70 mph
   - Distance between cities: 800 miles
   - Trains are traveling towards each other

2. Calculate the time difference between departures:
   - Time difference = 3:00 PM - 2:00 PM = 1 hour

3. Calculate the distance Train 1 travels in 1 hour:
   - Distance = Speed × Time
   - Distance = 60 miles/hour × 1 hour = 60 miles

4. Calculate the remaining distance when Train 2 starts:
   - Remaining distance = 800 miles - 60 miles = 740 miles

5. Calculate the combined speed of both trains:
   - Combined speed = 60 mph + 70 mph = 130 mph

6. Calculate the time taken to cover the remaining distance:
   - Time = Distance ÷ Speed
   - Time = 740 miles ÷ 130 miles/hour = 5.69 hours ≈ 5 hours 41 minutes

7. Determine the meeting time:
   - Meeting time = 3:00 PM + 5 hours 41 minutes = 8:41 PM
"""

solution = "The two trains will meet at approximately 8:41 PM."

result = evaluate_word_problem_reasoning(word_problem, reasoning, solution)
print(result)