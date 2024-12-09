import copy
import random

import numpy as np


class Node:

    def __init__(self, type):
        self.type = type


class Connect:
    def __init__(self, node_from: Node, node_to: Node):
        self.node_from = node_from
        self.node_to = node_to


class Genome:
    def __init__(self):
        self.nodes = []
        self.connections = []
        self.counter_type = {"SOM-LF": 0, "SOM-RS": 0, "SOM-SP": 0, "PCK-SP": 0, "PCK-RS": 0, "PCK-LF": 0, "OTH-XX": 0}
        self.neuron_type_counter = {"SOM": 0, "PCK": 0, "OTH": 0}
        self.connections_type_counter = {("SOM", "SOM"): 0, ("SOM", "PCK"): 0, ("SOM", "OTH"): 0, ("PCK", "SOM"): 0,
                                         ("PCK", "OTH"): 0, ("PCK", "PCK"): 0, ("OTH", "SOM"): 0, ("OTH", "PCK"): 0,
                                         ("OTH", "OTH"): 0}
        self.types = ["SOM-LF", "SOM-RS", "SOM-SP", "PCK-SP", "PCK-RS", "PCK-LF", "OTH-XX", ]

    def init_genome(self, nodes: [], connections: []):
        self.nodes = nodes
        self.connections = connections

    def add_node(self, type):
        self.counter_type[type] += 1
        self.neuron_type_counter[type[0:3]] += 1
        node = Node(type)
        self.nodes.append(node)
        return node

    def add_random_node(self):
        return self.add_node(random.choice(self.types))

    def add_random_connection(self):
        node1 = random.choice(self.nodes)
        node2 = random.choice(self.nodes)
        while node1 == node2:
            node2 = random.choice(self.nodes)
        self.connections.append(Connect(node1, node2))
        self.update_counter_connection_type(node1.type, node2.type)

    def update_counter_type(self, type):
        self.connections_type_counter[type] += 1

    def update_counter_connection_type(self, type: str, type2: str):
        self.connections_type_counter[(type[0:3], type2[0:3])] += 1

    def subtract_counter_type(self, type):
        self.connections_type_counter[type] -= 1

    def subtract_counter_connection_type(self, type: str, type2: str):
        self.connections_type_counter[(type[0:3], type2[0:3])] -= 1

    def add_connection(self, node_to, node_from):
        self.update_counter_connection_type(node_from.type, node_to.type)
        self.connections.append(Connect(node_from, node_to))

    def get_all_connection_by_node(self, node: Node):
        list_connection = []
        for connection in self.connections:
            if connection.node_to == node or connection.node_from:
                list_connection.append(connection)
        return list_connection

    def add_between_connection(self, node: Node, connection: Connect):
        if random.random() > 0.5:
            old_node_to = connection.node_to
            connection.node_to = node
            self.add_connection(node, old_node_to)
        else:
            old_node_from = connection.node_from
            connection.old_node_from = node
            self.add_connection(old_node_from, node)

    # creates a node between connections
    def mutate_add_node(self):
        if len(self.connections) == 0:
            return

        node = self.add_random_node()
        connection = random.choice(self.connections)
        self.add_between_connection(node, connection)

    def remove_node(self, node):
        self.subtract_counter_type(node.type)
        self.nodes.remove(node)

    def remove_connection(self, connection):
        self.subtract_counter_connection_type(connection.node_from.type, connection.node_to.type)
        self.connections.remove(connection)

    def mutate_remove_node(self):
        if len(self.nodes) == 0:
            return
        node = random.choice(self.nodes)
        connections_list = self.get_all_connection_by_node(node)

        for connection in connections_list:
            self.remove_connection(connection)

    # creates a connection between two not connected node
    def mutate_add_connection(self):
        if len(self.nodes) == 0:
            return

        node_1 = random.choice(self.nodes)
        node_2 = random.choice(self.nodes)

        while node_1 == node_2:
            node_2 = random.choice(self.nodes)

        self.add_connection(node_1, node_2)

    def mutate_remove_connection(self):
        if len(self.connections) == 0:
            return
        connection = random.choice(self.connections)
        self.remove_connection(connection)

    def select_genome_subset(self):
        connections_subset = random.sample(self.connections, random.randint(0, len(self.connections)))
        node_subset = set()
        for connection in connections_subset:
            node_subset.add(connection.node_to)
            node_subset.add(connection.node_from)

        return list(node_subset), connections_subset

    def connection_pck_som_mean(self):
        if len(self.connections) > 0:
            return self.connections_type_counter[("PCK", "SOM")] / len(self.connections)
        return 0
    def connection_mean(self):
        if len(self.nodes) > 0:
            return len(self.connections) / len(self.nodes)
        return 0
    def fitness_node_counter(self):
        return len(self.nodes)

    def copy(self):
        # Create a new genome object
        new_genome = Genome()

        # Copy the nodes, connections, and other attributes
        new_genome.nodes = copy.deepcopy(self.nodes)  # Deep copy the nodes (assuming Node has copy functionality)
        new_genome.connections = copy.deepcopy(self.connections)  # Deep copy the connections
        new_genome.counter_type = self.counter_type.copy()  # Shallow copy of the dictionary
        new_genome.neuron_type_counter = self.neuron_type_counter.copy()  # Shallow copy of the dictionary
        new_genome.connections_type_counter = self.connections_type_counter.copy()  # Shallow copy of the dictionary

        return new_genome


# GENERATING RANDOM GRAPH BASED ON NEAT EVOLUTION USING MULTIPLE OBJECT OPTIMIZATION

class Individual:
    def __init__(self, genome: Genome):
        self.genome = genome
        self.fitness = []
        self.crowding_distance = 0
        self.fitness_1 = 0
        self.fitness_2 = 0
        self.fitness_3 = 0
        self.rank = 0

    def get_genome(self):
        return self.genome

    def evaluate_fitness(self):
        self.fitness_connection_som_som_mean()
        self.fitness_connection_mean()
        self.fitness_node_counter()
        self.fitness_connection_pck_som_mean()

    def fitness_connection_som_som_mean(self):
        return 0

    def fitness_connection_pck_som_mean(self):
        self.fitness_1 = (0.1 - self.genome.connection_pck_som_mean()) ** 2

    def fitness_connection_mean(self):
        self.fitness_2 = (3.5 - self.genome.connection_mean()) ** 2

    def fitness_node_counter(self):
        self.fitness_3 = abs(1680 - self.genome.fitness_node_counter())

    def get_fitness_vector(self):
        return [self.fitness_1, self.fitness_2, self.fitness_3]

    def copy(self):
        # Create a new individual object with a copy of the genome
        new_individual = Individual(self.genome.copy())  # Copy the genome as well
        new_individual.fitness = self.fitness.copy()  # Copy the fitness list
        new_individual.crowding_distance = self.crowding_distance
        new_individual.fitness_1 = self.fitness_1
        new_individual.fitness_2 = self.fitness_2
        new_individual.fitness_3 = self.fitness_3
        new_individual.rank = self.rank
        return new_individual


def crowding_distance_calculation(population: [], comparator: callable, evaluate_crowding_distance: callable,
                                  fitness_counter) -> []:
    population = sorted(population, key=lambda x: x.get_fitness_vector(), reverse=True)

    crowding_distance = [np.inf] * len(population)

    if len(population) < 3:
        return crowding_distance
    # calculate crowding distance for weighted selecting

    max_fitness = [0.0] * fitness_counter
    min_fitness = [np.inf] * fitness_counter

    for i in range(len(population)):
        for j in range(fitness_counter):
            max_fitness[j] = max(max_fitness[j], population[i].get_fitness_vector()[j])
            min_fitness[j] = min(min_fitness[j], population[i].get_fitness_vector()[j])

    for i in range(1, len(population) - 1):
        crowding_distance[i] = 0
        for j in range(fitness_counter):
            if max_fitness[j] - min_fitness[j] == 0:
                crowding_distance[i] += 0
            else:
                crowding_distance[i] += evaluate_crowding_distance(population[i - 1], population[i + 1], j) / (max_fitness[
                                                                                                               j] -
                                                                                                           min_fitness[
                                                                                                               j])

    return crowding_distance


def fast_non_dominated_sorting(population: [], comparator: callable):
    front = [[]]

    dominated_counter = [[] for _ in population]
    dominator_counter = [0] * len(population)

    for i in range(len(population)):
        for j in range(len(population)):
            if i != j:
                if comparator(population[i], population[j]):
                    dominated_counter[i].append(j)  # i dominates j
                elif comparator(population[j], population[i]):
                    dominator_counter[i] += 1  # j dominates i

        if dominator_counter[i] == 0:  # No one dominates i
            front[0].append(i)  # Add index i to front[0]
            population[i].rank = 0
    i = 0
    while front[i]:
        Q = []
        for p in front[i]:  # p is an index
            for q in dominated_counter[p]:  # q is an index
                dominator_counter[q] -= 1
                if dominator_counter[q] == 0 and q not in Q:
                    Q.append(q)
                    population[q].rank = i
        i += 1
        front.append(Q)

    del front[-1]  # Remove last empty front
    return [[population[i] for i in f] for f in front]  # Convert indices back to objects


def crossover(parent_1: Individual, parent_2: Individual):
    node_subset_1, connection_subset_1 = parent_2.get_genome().select_genome_subset()
    node_subset_2, connection_subset_2 = parent_1.get_genome().select_genome_subset()
    new_node_list = node_subset_1 + node_subset_2
    new_connection_subset = connection_subset_1 + connection_subset_2
    new_genome = Genome()
    new_genome.init_genome(new_node_list, new_connection_subset)

    return Individual(new_genome)


def binary_selection_tournament(population):
    individual_1 = random.choice(population)
    individual_2 = random.choice(population)

    # Selection based on rank (lower is better)
    if individual_1.rank < individual_2.rank:
        return individual_1
    elif individual_1.rank > individual_2.rank:
        return individual_2

    # If ranks are equal, use crowding distance (higher is better)
    if individual_1.crowding_distance > individual_2.crowding_distance:
        return individual_1
    else:
        return individual_2


def mutation(individual: Individual):
    rnd = random.random()
    if rnd < 0.25:
        individual.genome.mutate_add_node()
    elif rnd < 0.5:
        individual.genome.mutate_remove_node()
    elif rnd < 0.75:
        individual.genome.mutate_add_connection()
    elif rnd < 1:
        individual.genome.mutate_remove_connection()


def init_population(create_random_solution: callable, population_size=1000) -> []:
    population = []
    for _ in range(population_size):
        sol = create_random_solution()
        population.append(sol)
    return population


def nsga_2(create_random_solution: callable, compare_individual: callable, evaluate_crowding_distance: callable,
           population_size=1000, max_population=1500, crossover_p=0.6,
           mutation_p=0.1, T_max=100):
    """
    Implements the NSGA-II algorithm.
    """
    population = init_population(population_size=population_size, create_random_solution=create_random_solution)
    for individual in population:
        individual.evaluate_fitness()

    for _ in range(T_max):  # Loop for T_max generations
        # 1. Perform Fast Non-Dominated Sorting
        front = fast_non_dominated_sorting(population=population, comparator=compare_individual)

        # 2. Calculate Crowding Distance
        for f in front:
            crowding_distance_calculation(f, compare_individual, evaluate_crowding_distance,
                                          len(population[0].get_fitness_vector()))

        # 3. Selection for the next generation
        next_population = []
        i = 0
        while i < len(front) and len(next_population) + len(front[i]) <= population_size:
            next_population.extend(front[i])
            i += 1

        if len(next_population) < population_size:
            # Sort the current front by crowding distance in descending order
            remaining = sorted(front[i], key=lambda x: x.crowding_distance, reverse=True)
            next_population.extend(remaining[:population_size - len(next_population)])

        # 4. Create Offspring Population via Crossover and Mutation
        off_spring = []
        while len(off_spring) + len(next_population) < max_population:
            parent1 = binary_selection_tournament(next_population)
            parent2 = binary_selection_tournament(next_population)

            # Apply crossover
            if random.random() < crossover_p:
                child_1 = crossover(parent1, parent2)
                off_spring.extend([child_1])
            else:
                # If no crossover, just clone the parents
                off_spring.extend([parent1.copy(), parent2.copy()])

            # Apply mutation
            for child in off_spring[-1:]:  # Mutate the last two added offspring
                if random.random() < mutation_p:
                    mutation(child)

        # 5. Evaluate fitness of the offspring
        for individual in off_spring:
            individual.evaluate_fitness()

        # 6. Combine populations and re-sort
        population = next_population + off_spring

    return fast_non_dominated_sorting(population=population, comparator=compare_individual)[0]


# ------------- INIT

def random_graph():
    genome = Genome()

    rnd = random.randint(100, 400)
    for _ in range(rnd):
        genome.add_random_node()

    rnd = random.randint(200, 800)
    for _ in range(rnd):
        genome.add_random_connection()

    return Individual(genome)


def compare_individual(individual_1, individual_2):
    fitness_vector_1 = individual_1.get_fitness_vector()
    fitness_vector_2 = individual_2.get_fitness_vector()

    is_strictly_better = False

    for i in range(len(fitness_vector_1)):
        if fitness_vector_1[i] < fitness_vector_2[i]:
            return False  # Not dominating
        elif fitness_vector_1[i] > fitness_vector_2[i]:
            is_strictly_better = True

    return is_strictly_better  # Dominates if strictly better in at least one objective


def evaluating_crowding_distance(individual_1: Individual, individual_2: Individual, i):
    return individual_2.get_fitness_vector()[i] - individual_1.get_fitness_vector()[i]


population = nsga_2(random_graph, compare_individual, evaluating_crowding_distance,T_max=20)
print(population[0].get_fitness_vector())


