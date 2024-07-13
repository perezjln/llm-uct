import numpy as np
import tqdm
from anytree import Node as RenderTree
from concurrent.futures import ThreadPoolExecutor, as_completed


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
def ollama_chat(model, current_solution=""):
    
    print(f"ollama_chat: current_solution: {current_solution}")
    
    response = ollama.chat(model=model, messages=[
        {
            'role': 'system',
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
            'role': 'system',
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

    def ucb(self, exploration_constant=np.sqrt(2), reward_weight=1.0, visit_weight=1.0):
        if self.parent is None:
            return float('inf')
        else:
            exploitation = (self.reward * reward_weight) / (self.visits + 1e-9)
            exploration = exploration_constant * np.sqrt(np.log(self.parent.visits + 1) / (self.visits * visit_weight + 1e-9))
            return exploitation + exploration



class UCT:
    def __init__(self, exploration_constant=np.sqrt(2), num_threads=4):
        self.exploration_constant = exploration_constant
        self.num_threads = num_threads
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

    def run_iteration(self):
        node = self.select(self.root)
        new_node = self.expand(node)
        reward = self.evaluate_node(new_node)
        self.backpropagate(new_node, reward)

    def uct(self, max_iterations):
        self.root = Node()

        with ThreadPoolExecutor(max_workers=self.num_threads) as executor:
            futures = [executor.submit(self.run_iteration) for _ in range(max_iterations)]
            for _ in tqdm.tqdm(as_completed(futures), total=max_iterations, desc="Running UCT: "):
                pass

        return max(self.root.children, key=lambda child: child.visits).current_solution

    def print_anytree(self):
        print(RenderTree(self.root))  # Print the tree to the console



def main():

    # Create a UCT agent
    uct = UCT(num_threads=4)

    # Run the UCT algorithm for 1000 iterations
    current_solution = uct.uct(max_iterations=10)

    # Print the best action
    print(f"The best solution is: {current_solution}")
    
    uct.print_anytree()

if __name__ == "__main__":
    main()
