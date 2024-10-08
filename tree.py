import random
from typing import List, Optional
import os
from openai import OpenAI
from dotenv import load_dotenv
import instructor
from pydantic import BaseModel
from collections import deque
import weave
from compare import real_llm_judge


# Updated MathNode Class
class MathNode:
    def __init__(self, task: str, task_thought_process: str = None, parent: Optional['MathNode'] = None, maintask: str = None):
        self.task = task  # Task description
        self.task_thought_process = task_thought_process  # Thought process used to generate the task
        self.solution: Optional[str] = None  # Sub-answer after solving the task
        self.solution_thought_process: Optional[str] = None  # Thought process for achieving the sub-answer
        self.parent = parent  # Reference to parent node
        self.children: List['MathNode'] = []  # List of child nodes
        self.isLeaf = False  # Flag to indicate if this node has reached a final answer (leaf node)
        
        # Store the main task explicitly, passing it down from the parent node
        self.maintask = maintask if maintask else (parent.maintask if parent else None)

# Pydantic models to parse the model's response
class Reasoning(BaseModel):
    way1: str
    thought_process1: str
    way2: Optional[str] = ""
    thought_process2: Optional[str] = ""


class Solution(BaseModel):
    sub_answer: str  # Intermediate answer for the current step/subtask
    final_answer: Optional[str] = None  # Final answer that terminates the reasoning, if available
    thought_process: str  # Thought process used to derive the sub_answer or final_answer


# Load environment variables
load_dotenv()
os.environ['OPENROUTER_API_KEY'] = os.getenv('OPENROUTER_API_KEY')

weave.init('together-weave')

class OpenAIModel:
    def __init__(self):
        self.client = instructor.from_openai(OpenAI(
            api_key=os.environ["OPENROUTER_API_KEY"],
            base_url="https://openrouter.ai/api/v1",
        ))

    @weave.op()
    def generate_steps(self, task: str, maintask: str, solution: str, task_thought_process: str, solution_thought_process: str, previous_tasks: List[str], previous_solutions: List[str], num_steps: int = 2) -> List[dict]:
        print(f"Generating {num_steps} alternative steps for the task: '{task}'")

        # Create context string from previous tasks and solutions
        context = " -> ".join([f"{t} (solution: {s})" for t, s in zip(previous_tasks, previous_solutions)])

        # Enhanced prompt for generating alternative distinct reasoning paths
        prompt = (
            f"From the current step: '{task}' (solution: '{solution}'), "
            f"with the thought process: '{task_thought_process}', and the solution reasoning: '{solution_thought_process}', "
            f"considering the main task: '{maintask}', and the previous tasks and solutions: {context}, "
            f"generate {num_steps} distinct alternative ways and thought processes to continue solving the main task based on the current step. "
            f"\n\n"
            f"These alternative ways must be different from the current step and each other."
            f"Each approach should explain why this particular strategy or next step is helpful in relation to the overall goal of solving the main task. "
            f"Provide clear reasoning (thought_process) for each step. "
            f"\n\n"
            f"Ensure that each approach follows a different strategy from the others, be creative."
            f"There must be **{num_steps} distinct alternative ways generated**, else the response will be invalid."
            f"\n\n"
            f"Return the response in the following format:\n"
            f"{{'way1': '...', 'thought_process1': '...'}}\n"
            f"{{'way2': '...', 'thought_process2': '...'}}"
        )


        # Request response from the model
        response = self.client.chat.completions.create(
            model="openai/gpt-4o",
            response_model=Reasoning,
            messages=[
                {"role": "system", "content": "You are a helpful assistant that generates distinct alternative steps for solving math problems. In this task, you must produce 2 alternative ways to continue solving the main task based on the current step."},
                {"role": "user", "content": prompt}
            ]
        )

        print("Generated steps response: ", response)

        # Parse the response into a list of step and thought_process pairings
        steps_and_reasonings = []

        steps_and_reasonings.append({
            'step': response.way1,
            'thought_process': response.thought_process1
        })

        steps_and_reasonings.append({
            'step': response.way2,
            'thought_process': response.thought_process2
        })

        return steps_and_reasonings


    @weave.op()
    def solve_step(self, step: str, previous_tasks: List[str], previous_solutions: List[str], maintask: str) -> dict:
        """
        Solves the current step by providing a sub-answer and checks if there is enough information to solve the main task.
        If the main task is solvable, it provides the final answer.
        """
        # Create context string from previous tasks and solutions
        context = " -> ".join([f"{t} (solution: {s})" for t, s in zip(previous_tasks, previous_solutions)])

        # Improved prompt for solving the step and checking if the main task can be solved
        prompt = (
            f"Solve the following step: '{step}'. "
            f"Considering the previous tasks and solutions: {context}, "
            f"provide the sub-answer for this step with a thought process explaining the reasoning behind it. "
            f"Then, check if you have enough information to solve the main task: '{maintask}'. If yes, provide the final answer for the main task. "
            f"If not, return the sub-answer."
        )

        # Request response from the model with a structured response model (Solution)
        response = self.client.chat.completions.create(
            model="openai/gpt-3.5-turbo",
            response_model=Solution,
            messages=[
                {"role": "system", "content": "You are a helpful assistant that solves steps in math problems."},
                {"role": "user", "content": prompt}
            ]
        )

        print("Solve step response: ", response)

        # Check if a final answer is provided
        if response.final_answer:
            # If the final answer is provided, no further steps are required (leaf node)
            return {
                'sub_answer': response.sub_answer,
                'thought_process': response.thought_process,
                'final_answer': response.final_answer,  # Final answer means this path is complete
            }
        else:
            # If no final answer, provide the sub-answer and indicate that further steps are needed
            return {
                'sub_answer': response.sub_answer,
                'thought_process': response.thought_process,
                'final_answer': "",  # No final answer, further steps are needed
            }


class MathTree:
    MAX_DEPTH = 1

    def __init__(self, initial_task: str, model: OpenAIModel):
        self.root = MathNode(task=initial_task, maintask=initial_task)
        self.model = model
        self.solution_paths = []  # Store valid solution paths
        self.queue = deque([self.root])  # Initialize a queue for BFS-style search

    def bfs_search(self):
        """Perform BFS-style MCTS search to find multiple solution paths."""
        while self.queue:
            # Dequeue the next node for exploration
            node = self.queue.popleft()

            # Check if the node exceeds max depth
            depth = self.get_node_depth(node)
            if depth > self.MAX_DEPTH:
                continue  # Skip nodes that exceed max depth

            # Expand the node and add its children to the queue
            expanded_node = self.expand(node)
            self.queue.extend(expanded_node.children)

            # Collect the solution path if we reach a solution
            for child in expanded_node.children:
                self.collect_solution_path(child)

    def expand(self, node: MathNode) -> MathNode:
        """Expand a node by generating and solving the frontier (possible subtasks)."""

        # If the node is a leaf (final answer reached), no further expansion is needed
        if node.isLeaf:
            print(f"Node is already a leaf with final answer: {node.solution}, no further expansion.")
            return node

        previous_tasks, previous_solutions = self.get_context_from_node(node)

        # Generate the next set of distinct reasoning paths based on the current subtask and solution
        steps = self.model.generate_steps(
            task=node.task,  # Use the current subtask as the task to generate the next steps
            maintask=node.maintask,  # Include the main task for context
            solution=node.solution,  # Pass the solution to the current task
            task_thought_process=node.task_thought_process,  # Include the thought process behind the current subtask
            solution_thought_process=node.solution_thought_process,  # Include the thought process behind solving it
            previous_tasks=previous_tasks, 
            previous_solutions=previous_solutions, 
            num_steps=2
        )

        if not steps:
            return node  # Return the node as-is if no steps were generated

        # Add generated steps as child nodes
        for step in steps:
            # Create a child node for each step and track thought process
            child_node = MathNode(
                task=step['step'],
                task_thought_process=step['thought_process'],
                parent=node,
                maintask=node.maintask
            )
            node.children.append(child_node)

        # Solve each unsolved step and update node information
        for child_node in node.children:
            # Solve the subtask with the context of the parent task and previous steps
            solution = self.model.solve_step(
                step=child_node.task,
                previous_tasks=previous_tasks,
                previous_solutions=previous_solutions,
                maintask=node.maintask  # Pass the main task explicitly
            )

            # Store the sub-answer and the thought process for this subtask
            child_node.solution = solution['sub_answer']
            child_node.solution_thought_process = solution['thought_process']

            # Check if there's a final answer to terminate the path
            if solution.get('final_answer'):
                child_node.isLeaf = True  # Mark this node as a leaf
                child_node.solution = solution['final_answer']
                child_node.solution_thought_process = solution['thought_process']
                print(f"Final answer reached: {child_node.solution}")

        return node


    def get_node_depth(self, node: MathNode) -> int:
        """Calculate the depth of a node from the root."""
        depth = 0
        current = node
        while current.parent:
            depth += 1
            current = current.parent
        return depth

    def collect_solution_path(self, node: MathNode, for_judge: bool = False) -> List[dict]:
        """Collect a valid solution path up to the current node."""
        path = []
        current_node = node

        while current_node != self.root:
            step_details = {
                'task': current_node.task,
                'task_thought_process': current_node.task_thought_process,
                'sub_answer': current_node.solution,  # This can be either sub-answer or final answer
                'answer_thought_process': current_node.solution_thought_process
            }
            path.append(step_details)
            current_node = current_node.parent

        path.reverse()  # Reverse to get the correct order from root to leaf

        if for_judge:
            # Simplified solution path for the judge: only include task and sub-answer
            simplified_path = [{'task': step['task'], 'sub_answer': step['sub_answer']} for step in path]
            return simplified_path

        # Full path for general collection
        self.solution_paths.append(path)
        return path


    @weave.op()
    def solve_task(self) -> List[dict]:
        """
        Solve the task by performing BFS-style MCTS search, comparing step pairs, and returning detailed DPO results.

        Output:
        - List of JSON-like dicts with:
        - path_up_to_steps
        - step1 details (task, sub_answer, task_thought_process)
        - step2 details (task, sub_answer, task_thought_process)
        - llm_judge_winner
        - llm_judge_thought_process
        """
        self.bfs_search()  # Start the BFS search

        dpo_pairs = []  # To store DPO evaluation results

        # Initialize the BFS queue with the root node
        queue = deque([self.root])

        while queue:
            node = queue.popleft()  # Get the next node in the BFS queue

            # If this node has at least two children, we can compare them
            if len(node.children) >= 2:
                step1, step2 = node.children[0], node.children[1]

                # Collect the solution path leading up to the two steps (this path will be the same for both steps)
                path_up_to_steps = self.collect_solution_path(node, for_judge=True)
                print(f"Path up to steps: {path_up_to_steps}")

                # Collect subtask generation and thought process for each step
                step1_details = {
                    'task': step1.task,
                    'thought_process': step1.task_thought_process,
                }
                step2_details = {
                    'task': step2.task,
                    'thought_process': step2.task_thought_process,
                }

                # Prepare inputs for LLM judge
                path_up_till_current_step = " -> ".join([step['task'] for step in path_up_to_steps])
                
                judge_decision = real_llm_judge(
                    path_up_till_current_step=path_up_till_current_step,
                    way_1_subtask=step1.task,
                    way_1_task_thought_process=step1.task_thought_process,
                    way_2_subtask=step2.task,
                    way_2_task_thought_process=step2.task_thought_process
                )

                # Collect everything into a single DPO result
                dpo_result = {
                    'path_up_to_steps': path_up_to_steps,  # Same path for both steps
                    'step1': step1_details,
                    'step2': step2_details,
                    'llm_judge_winner': judge_decision['winner'],
                    'llm_judge_thought_process': judge_decision['reasoning'],
                }

                # Append the DPO result for this pair of steps
                dpo_pairs.append(dpo_result)

            # Continue BFS by adding children to the queue
            for child in node.children:
                queue.append(child)

        return dpo_pairs


    def get_context_from_node(self, node: MathNode):
        """Extract the full path of tasks and solutions up to this node."""
        tasks = []
        solutions = []
        current = node
        while current.parent:
            tasks.append(current.parent.task)
            solutions.append(current.task)
            current = current.parent
        return tasks[::-1], solutions[::-1]
    
    
    def run_llm_judge(self, path_up_till_current_step: str, way_1_subtask: str, way_1_task_thought_process: str, way_2_subtask: str, way_2_task_thought_process: str) -> dict:
        """
        A dummy LLM judge function that takes in two alternative steps and their thought processes,
        and returns a JSON-like response with the winner and thought process.
        
        Input:
        - path_up_till_current_step: str
        - way_1_subtask: str
        - way_1_task_thought_process: str
        - way_2_subtask: str
        - way_2_task_thought_process: str
        
        Output:
        - JSON-like dict with 'winner' and 'thought_process'.
        """
        # Placeholder logic to simulate judging the two steps
        # For demonstration, we'll randomly select "way_1" or "way_2" as the winner
        winner = "way_1" if random.choice([True, False]) else "way_2"

        # Dummy thought process for the LLM judge's decision
        judge_thought_process = (
            f"Based on the evaluation of both approaches, '{winner}' was chosen because "
            f"it offers a more direct or efficient path to solving the task. "
            f"The thought process of '{winner}' aligns better with achieving the main objective."
        )

        # Return the winner and the thought process as a dictionary
        return {
            'winner': winner,
            'thought_process': judge_thought_process
        }
