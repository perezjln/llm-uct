import numpy as np
import tqdm
from anytree import Node as RenderTree

from problems import prompt_question_1, prompt_question_2, problem_remi


prompt_improvement= """Current solution (can be empty): [{current_solution}].
Task: Improve this solution.
Be short and concise.
"""

prompt_evaluate= """ Current solution: {current_solution}.
Task: Rate the solution on a scale of 1 to 10.
ONLY PROVIDE THE RATING.
"""

model_id = "gemma2:9b-instruct-q2_K"

import ollama
def ollama_chat(model, prompt_question, current_solution=""):
    
    print(f"ollama_chat: current_solution: {current_solution}")
    
    response = ollama.chat(model=model, messages=[
        {
            'role': 'system',
            'content': prompt_question,
        },
        {
            'role': 'user',
            'content': prompt_improvement.format(current_solution=current_solution),
        },
    ])
    print(f"ollama_chat: response: {response['message']['content']}")
    return response['message']['content']


def ollama_evaluate(model, prompt_question, current_solution=""):
    
    print(f"ollama_evaluate: current_solution: {current_solution}")
    
    response = ollama.chat(model=model, messages=[
        {
            'role': 'system',
            'content': prompt_question,
        },
        {
            'role': 'user',
            'content': prompt_evaluate.format(current_solution=current_solution),
        },
    ])
    print(f"ollama_evaluate: response: {response['message']['content']}")
    return response['message']['content']


class Node:
    def __init__(self, parent=None):
        self.parent = parent
        self.current_solution = ""
        self.children = []
        self.visits = 0
        self.reward = 0

    def __repr__(self):
        return f"Node(visits={self.visits}, reward={self.reward}, current_solution={self.current_solution})"

    def ucb(self, exploration_constant=np.sqrt(2), reward_weight=1.0, visit_weight=1.0):
        if self.parent is None:
            return float('inf')
        else:
            exploitation = (self.reward * reward_weight) / (self.visits + 1e-9)
            exploration = exploration_constant * np.sqrt(np.log(self.parent.visits + 1) / (self.visits * visit_weight + 1e-9))
            return exploitation + exploration


class UCT:
    
    def __init__(self, prompt, exploration_constant=np.sqrt(2)):
        self.exploration_constant = exploration_constant
        self.root = None
        self.prompt = prompt

    def select(self, node):
        while node.children:
            node = max(node.children, key=lambda child: child.ucb(self.exploration_constant))
        return node

    def expand(self, node):
        child = Node(parent=node)
        child.current_solution = ollama_chat(model=model_id, prompt_question=self.prompt, current_solution=node.current_solution)
        node.children.append(child)
        return child

    def evaluate_node(self, node):
        return float(ollama_evaluate(model=model_id, prompt_question=self.prompt, current_solution=node.current_solution))

    def backpropagate(self, node, reward):
        while node:
            node.visits += 1
            node.reward += reward
            node = node.parent

    def uct(self, max_iterations):
        self.root = Node()
        for _ in tqdm.tqdm(range(max_iterations), desc="Running UCT: "):
            node = self.select(self.root)
            new_node = self.expand(node)
            reward = self.evaluate_node(new_node)
            self.backpropagate(new_node, reward)

        return max(self.root.children, key=lambda child: child.visits).current_solution    
        
    def print_anytree(self):
        print(RenderTree(self.root))  # Print the tree to the console


def main():

    # Create a UCT agent
    uct = UCT(prompt=problem_remi)

    # Run the UCT algorithm for 1000 iterations
    current_solution = uct.uct(max_iterations=10)

    # Print the best action
    print(f"The best solution is: {current_solution}")
    
    uct.print_anytree()

if __name__ == "__main__":
    main()
