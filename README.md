##Generating a Random Graph Using Multi-Objective Optimization##
**Overview**
This project aims to generate a neural network graph that adheres to predefined constraints, such as:
- Number of nodes
- Connection type rate mean
- Total connection mean
- ...
To achieve this, an evolutionary algorithm based on NSGA-II (Non-dominated Sorting Genetic Algorithm II) is implemented. The algorithm evolves an initial population of graphs to converge towards statistical indices provided as constraints.

**Objectives**
The main objective is to construct a neural network graph that simulates a specific structure, for instance, a brain neural network related to pain simulation, while respecting various constraints.

**Workflow**
Initial Population Creation

The initial graph population is generated using the Erdős–Rényi algorithm, which produces random graphs by connecting nodes with a given probability.
A fitness function is defined to evaluate how well each graph respects the provided constraints.
**Optimization via NSGA-II**

The standard NSGA-II framework is applied to optimize the graph structure.
A specialized crossover mechanism is implemented, inspired by evolutionary neural networks.
This involves performing connections crossover across graphs with different structures.
Mutation and selection operators are used to maintain diversity and guide the population toward convergence.
**Termination**

The algorithm ends when convergence criteria are met, producing graphs that satisfy the desired statistical properties.
Real-World Application
The project includes a real-world example:

Simulating a brain neural network for pain modeling.
The generated graph structures can mimic connections observed in neural systems, offering potential insights into pain-related pathways.
**Key Components**
Graph Generation: Erdős–Rényi random graph initialization.
Fitness Evaluation: Ensures compliance with statistical constraints.
NSGA-II Optimization: Multi-objective optimization to improve the graph population.
Crossover and Mutation: Evolutionary techniques for neural network-like graph structures.
