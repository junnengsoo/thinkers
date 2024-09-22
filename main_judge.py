import os
import instructor
from openai import OpenAI
import weave
from dotenv import load_dotenv
from typing import Dict


load_dotenv()

weave.init('together-weave')

os.environ['OPENROUTER_API_KEY'] = os.getenv('OPENROUTER_API_KEY')

client = OpenAI(
    api_key=os.environ["OPENROUTER_API_KEY"],
    base_url="https://openrouter.ai/api/v1",
)

SYSTEM_PROMPT = """

Evaluate the two approaches based on logical reasoning and computational efficiency.

"""

def main_judge(
    path_up_till_current_step: str,
    way_1_subtask: str,
    way_1_task_thought_process: str,
    way_2_subtask: str,
    way_2_task_thought_process: str
) -> Dict[str, str]:
    """
    Evaluates two different ways to accomplish a subtask based on a predefined system prompt and returns
    the winner along with the reasoning for the preference.

    Args:
    path_up_till_current_step (str): The path and steps taken up until the current decision point.
    way_1_subtask (str): Description of the first subtask or approach.
    way_1_task_thought_process (str): Thought process behind the first subtask approach.
    way_2_subtask (str): Description of the second subtask or approach.
    way_2_task_thought_process (str): Thought process behind the second subtask approach.

    Returns:
    dict: JSON object with the winner ("way_1" or "way_2") and the thought process for the preference.
    """
    
    SYSTEM_PROMPT = """
    
    You are tasked with evaluating two different approaches to solving a problem. 
    Consider each approach objectively based on its merits and how effectively it addresses the given task. 
    Factors to evaluate include logical reasoning, efficiency, maintainability, and relevance to the current context. 
    Select the approach that best addresses the task at hand and provide clear reasoning for your choice.
    
    The output should be in json format with the following structure:
    {
        "winner": "way_1" or "way_2",
        "reasoning": "Reasoning for selecting way_1 or way_2."
    } 
    """

    # Format the input for the model's evaluation
    reasoning_text = f"""
    Path up till current step: {path_up_till_current_step}

    Way 1 Subtask: {way_1_subtask}
    Way 1 Thought Process: {way_1_task_thought_process}

    Way 2 Subtask: {way_2_subtask}
    Way 2 Thought Process: {way_2_task_thought_process}
    """

    try:
        chat_completion = client.chat.completions.create(
            model="openai/gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": reasoning_text},
            ],
            temperature=0.7,
            max_tokens=1024,
        )
        response = chat_completion.choices[0].message.content
        # print("Model response 2:", response)
        return response
        
    except Exception as e:
        print("An error occurred during the evaluation:", e)
        return {"error": str(e)}

# Example Test Case
result = main_judge(
    path_up_till_current_step="Solved equations using substitution but ran into complex fractions.",
    way_1_subtask="Simplify the fractions first before solving for variables.",
    way_1_task_thought_process="This method eliminates the fractions early on, making later steps easier, but requires extra manipulation at the start.",
    way_2_subtask="Solve for variables directly without simplifying the fractions.",
    way_2_task_thought_process="This method skips the fraction simplification, going straight to variable solutions, but makes the final steps more complex."
)

print(result)
