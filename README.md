# LLM_UCT

## Overview
LLM_UCT is a simple set of snippets that explore the Upper Confidence Bound for Trees (UCT) algorithm in conjunction with a large language model (LLM) to solve complex problems. 
In other words, UCT is used as a discourse planner or reasoning strategy over an LLM output.
The project defines a Node class to represent states and uses UCT to navigate through potential solutions, leveraging the LLM for generating and evaluating responses.

## Features
- **Node Class**: Represents the state in the UCT algorithm with attributes for parent, children, visits, reward, and current solution.
- **UCT Algorithm**: Implements the UCT algorithm to explore and exploit potential solutions.
- **LLM Integration**: Utilizes a language model to generate and evaluate solutions during the UCT process.
- **Anytree Integration**: Visualizes the tree structure of explored nodes.

## Installation
1. Clone the repository:
    ```bash
    git clone https://github.com/perezjln/LLM_UCT.git
    cd LLM_UCT
    ```
2. Install the required packages:
    ```bash
    pip install numpy tqdm anytree ollama
    ```

## Usage
To run the UCT algorithm and find the best solution, execute the `main.py` script:
```bash
python main.py
```

### Example
Here is an example of how to use the UCT class:

1. Create a UCT agent.
2. Run the UCT algorithm for a specified number of iterations.
3. Print the best solution.
4. Visualize the tree structure.

```python
from uct import UCT

# Create a UCT agent
uct = UCT()

# Run the UCT algorithm for 1000 iterations
current_solution = uct.uct(max_iterations=1000)

# Print the best action
print(f"The best solution is: {current_solution}")

# Visualize the tree structure
uct.print_anytree()
```

## Code Explanation

### Node Class
The `Node` class represents a state in the UCT algorithm. It includes attributes for the parent node, children nodes, visit count, reward, and the current solution. The `ucb` method calculates the Upper Confidence Bound (UCB) value for the node, balancing exploration and exploitation.

### UCT Class
The `UCT` class implements the UCT algorithm. It includes methods for:
- Selecting the most promising node (`select`).
- Expanding the selected node by generating new solutions (`expand`).
- Evaluating the new solution using the LLM (`evaluate_node`).
- Backpropagating the results to update the tree (`backpropagate`).
- Running the UCT algorithm for a specified number of iterations (`uct`).
- Printing the tree structure (`print_anytree`).

### Main Script
The `main.py` script creates a UCT agent, runs the UCT algorithm, prints the best solution, and visualizes the tree structure.

## Contributing
1. Fork the repository.
2. Create a new branch (`git checkout -b feature-branch`).
3. Commit your changes (`git commit -am 'Add new feature'`).
4. Push to the branch (`git push origin feature-branch`).
5. Create a new Pull Request.

## License
This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

---

Feel free to modify the content to better suit your project's specifics.