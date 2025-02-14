
from datasets import load_dataset
import numpy as np
import tqdm
from anytree import Node as RenderTree
import ollama


prompt_improvement= """this is the current solution, it can be empty: [{current_solution}].
task: improve the current solution.
ALWAYS be short and concise.
"""

prompt_evaluate= """this is the current solution: {current_solution}.
task: evaluate the current solution on a scale of 1 to 10.
DO NOT ADD ANY OTHER ADDITIONAL INFORMATION.
ONLY PROVIDE THE RATING.
"""

class Node:
    def __init__(self, parent=None):
        self.parent = parent
        self.current_solution = ""
        self.children = []
        self.visits = 0
        self.reward = 0
        self.reward_indiv = 0

    def __repr__(self):
        return f"Node(visits={self.visits}, reward={self.reward}, current_solution={self.current_solution})"
    
    def ucb(self, exploration_constant=np.sqrt(2)):
        if self.parent is None:
            return float('inf')
        else:
            return self.reward / (self.visits + 1e-9) + exploration_constant * np.sqrt(np.log(self.parent.visits) / (self.visits + 1e-9))


class UCT:
    def __init__(self, prompt, max_child=4, exploration_constant=np.sqrt(2)):
        self.exploration_constant = exploration_constant
        self.root = None
        self.prompt = prompt
        self.max_child = max_child

    def select(self, node):
        while len(node.children) == self.max_child:
            node = max(node.children, key=lambda child: child.ucb(self.exploration_constant))
        return node

    def expand(self, node):
        child = Node(parent=node)
        child.current_solution = self.ollama_chat(model=model_id, prompt_question=self.prompt, current_solution=node.current_solution)
        node.children.append(child)
        return child

    def evaluate_node(self, node):
        return float(self.ollama_evaluate(model=model_id, prompt_question=self.prompt, current_solution=node.current_solution))

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
            new_node.reward_indiv = reward
            self.backpropagate(new_node, reward)

        return max(self.root.children, key=lambda child: child.visits).current_solution    

    ## Find the node with the best reward value by traversing the tree
    def find_best_node(self, node=None):
        if node is None:
            node = self.root

        best_node = node
        stack = [node]

        while stack:
            current_node = stack.pop()
            if current_node.reward_indiv > best_node.reward_indiv:
                best_node = current_node
            stack.extend(current_node.children)

        return best_node
        
    def print_anytree(self):
        print(RenderTree(self.root))  # Print the tree to the console
        
    @staticmethod
    def ollama_chat(model, prompt_question, current_solution="", do_log=False):
    
        if do_log:
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
        
        if do_log:
            print(f"ollama_chat: response: {response['message']['content']}")
        return response['message']['content'].strip()

    @staticmethod
    def ollama_evaluate(model,  prompt_question, current_solution="", do_log=False):
        
        if do_log:
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
        if do_log:
            print(f"ollama_evaluate: response: {response['message']['content']}")
            
        return float(response['message']['content'].strip())


def main():

    #data = load_dataset("lighteval/MATH", split="train", trust_remote_code=True)
    data = load_dataset("lighteval/mmlu", "philosophy", split="test", trust_remote_code=True)


    for prompt_question in data["question"]:
        
        print(f"Prompt: {prompt_question}")
        
        # Create a UCT agent
        uct = UCT(prompt_question)

        # Run the UCT algorithm for 1000 iterations
        uct.uct(max_iterations=10)
        best_node = uct.find_best_node()
        print(f"The best solution is {best_node.reward_indiv}: {best_node.current_solution}")
        


if __name__ == "__main__":
    model_id = "gemma2:9b-instruct-q2_K"    
    main()
