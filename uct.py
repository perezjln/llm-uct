import numpy as np
import tqdm
from anytree import Node as RenderTree


prompt_question_1 = """
You have a bag containing 3 red marbles and 5 blue marbles. 
You randomly draw 2 marbles from the bag without replacing them. 
What is the probability that both marbles are red?
be short and concise.
"""

prompt_question_2 = """
You have a standard deck of 52 playing cards. 
You randomly draw 3 cards from the deck without replacing them. 
What is the probability that all 3 cards are face cards (Jacks, Queens, or Kings)?
Be short and concise.
"""

prompt_improvement= """this is the current solution, it can be empty: [{current_solution}].
task: improve the current solution.
be short and concise.
"""

prompt_evaluate= """this is the current solution: {current_solution}.
task: evaluate the current solution on a scale of 1 to 10.
DO NOT ADD ANY OTHER ADDITIONAL INFORMATION.
ONLY PROVIDE THE RATING.
"""

model_id = "gemma2:9b-instruct-q2_K"

import ollama
def ollama_chat(model, current_solution=""):
    
    print(f"ollama_chat: current_solution: {current_solution}")
    
    response = ollama.chat(model=model, messages=[
        {
            'role': 'user',
            'content': prompt_question_2,
        },
        {
            'role': 'user',
            'content': prompt_improvement.format(current_solution=current_solution),
        },
    ])
    print(f"ollama_chat: response: {response['message']['content']}")
    return response['message']['content']


def ollama_evaluate(model, current_solution=""):
    
    print(f"ollama_evaluate: current_solution: {current_solution}")
    
    response = ollama.chat(model=model, messages=[
        {
            'role': 'user',
            'content': prompt_question_2,
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
    
    def ucb(self, exploration_constant=np.sqrt(2)):
        if self.parent is None:
            return float('inf')
        else:
            return self.reward / (self.visits + 1e-9) + exploration_constant * np.sqrt(np.log(self.parent.visits) / (self.visits + 1e-9))


class UCT:
    
    def __init__(self, exploration_constant=np.sqrt(2)):
        self.exploration_constant = exploration_constant
        self.root = None

    def select(self, node):
        while node.children:
            node = max(node.children, key=lambda child: child.ucb(self.exploration_constant))
        return node

    def expand(self, node):
        child = Node(parent=node)
        child.current_solution = ollama_chat(model=model_id, current_solution=node.current_solution)
        node.children.append(child)
        return child

    def evaluate_node(self, node):
        return float(ollama_evaluate(model=model_id, current_solution=node.current_solution))

    def backpropagate(self, node, reward):
        while node:
            node.visits += 1
            node.reward += reward
            node = node.parent

    def uct(self, max_iterations):
        self.root = Node()
        for _ in tqdm.tqdm(range(max_iterations)):
            node = self.select(self.root)
            new_node = self.expand(node)
            reward = self.evaluate_node(new_node)
            self.backpropagate(new_node, reward)

        return max(self.root.children, key=lambda child: child.visits).current_solution    
        
    def print_anytree(self):
        print(RenderTree(self.root))  # Print the tree to the console


def main():

    # Create a UCT agent
    uct = UCT()

    # Run the UCT algorithm for 1000 iterations
    current_solution = uct.uct(max_iterations=10)

    # Print the best action
    print(f"The best solution is: {current_solution}")
    
    uct.print_anytree()

if __name__ == "__main__":
    main()
