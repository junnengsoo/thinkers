Generate multiple task-solving trajectories, utilizing a modified Monte Carlo Tree Search (MCTS) to explore alternative reasoning paths for each step of a complex task. At each step, an LLM-as-a-judge evaluates the different reasoning strategies using Direct Preference Optimization (DPO), deciding which reasoning best contributes to solving the task. The LLM judge’s decisions, alongside its reasoning, are appended to output preferential datasets. These datasets are designed to help fine-tune smaller, less performant models, improving their ability to generate chain-of-thought reasoning and rationale.


Running the code
```
python app.py
```
