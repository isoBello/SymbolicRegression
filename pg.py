#!/usr/bin/env python3
from copy import deepcopy
from random import randint, random
import math
import logging
import numpy as np
import matplotlib.pyplot as plot
import sys

# Constants used to the experimental analysis of the PG
executions = 10
depth_individual = 7
alfa_crossover = 0.9
beta_mutation = 0.05
tournament_size = 2
population_size = 50
elite_individuals = 2
generations = 100
optimal_fitness = 0.0

# Variables used to present the results of the PG
best_individual = None
best_fitness = 0.0
worst_fitness = 0.0
average_fitness = 0.0
best_evolution = []
average_evolution = []
worst_evolution = []
best_children = []
last_generation = 0.0
crossover_count = 0
best_children_gen = []
best_individuals_crossover = []
number_data = 1

# Tree characteristics
operator = "Operator"
constant = "Constant"
variable = "Variable"
functions = {'add': 1, 'sub': 2, 'mul': 3, 'div': 4, 'sen': 5, 'coss': 6}
variables = {'x': 1}
variables_size = len(variables)
interval_constants = range(-10, 10)
functions_size = 4
depth_max = 6
depth_min = 2

# Individual characteristics
min_depth_individual = 1
max_depth_individual = 2
index_individual = 4


class Node:
    def __init__(self, node_type, value, height, leaf_left=None, leaf_right=None):
        self.type = node_type
        self.value = value
        self.height = height
        self.leaf_left = leaf_left
        if node_type != operator or value < 5:
            # Used in right leaves with sen or cos values, 'cause they need a constant after showed
            # (i.e we don't have sen + 2, just sen 2)
            self.leaf_right = leaf_right
        else:
            self.leaf_right = Node(constant, 0.0, height + 1)

    def evaluation(self, values):
        if self.type == constant:
            return self.value
        elif self.type == operator:
            return self.function_evaluation(self.value, self.leaf_left.evaluation(values),
                                            self.leaf_right.evaluation(values))
        elif self.type == variable:
            return values[self.value]

    @staticmethod
    def function_evaluation(opration, a, b):
        if opration == functions['add']:
            return a + b
        elif opration == functions['sub']:
            return a - b
        elif opration == functions['mul']:
            return a * b
        elif opration == functions['div']:
            return a / b if b else 0  # Prevents errors from division: num / 0
        elif opration == functions['sen']:
            return math.sin(a)
        elif opration == functions['coss']:
            return math.cos(b)

    def subtree(self, depth):
        if self.height == depth:  # If we asked a subtree who has the same depth and height
            return self
        else:
            side = randint(0, 1)  # 0 we going left-side; 1 we going right-side
            if side == 0:
                if self.leaf_left is not None:
                    return self.leaf_left.subtree(depth)
                else:
                    if self.leaf_right is not None:
                        return self.leaf_right.subtree(depth)
                    else:
                        return self
            else:
                if self.leaf_right is not None:
                    return self.leaf_right.subtree(depth)
                else:
                    if self.leaf_left is not None:
                        return self.leaf_left.subtree(depth)
                    else:
                        return self

    def tree_mutation(self, max_depth, index):
        type_mutation = randint(0, 2)
        if type_mutation == 0:  # Mutate into a variable
            self.type = variable
            self.value = rand_index(1, index)
            self.leaf_left = None
            self.leaf_right = None
        elif type_mutation == 1:  # Mutate into a constant
            self.type = constant
            self.value = rand_index(1, index)
            self.leaf_left = None
            self.leaf_right = None
        elif type_mutation == 2:  # Mutate into a function
            if self.height == max_depth:
                new_tree = self.change_subtree(depth_max, depth=self.height)
                Node.swap_subtree(self, new_tree)

    def swap_subtree(self, b):  # Swap between two nodes
        self.__dict__, b.__dict__ = b.__dict__, self.__dict__

    def print_tree(self, level):
        string_tree = ""
        if self.type == variable:
            string_tree += "x" + str(self.value)
        elif self.type == operator:
            string_tree += str(get_operator(self.value))
        else:
            string_tree += str(self.value)
        string_tree += "\n"
        if self.leaf_left is not None:
            string_tree += "Left Node" + str(self.leaf_left.height) + " -> " + self.leaf_left.print_tree(level + 1)
        if self.leaf_right is not None:
            string_tree += "Right Node" + str(self.leaf_right.height) + " -> " + self.leaf_right.print_tree(level + 1)


class Individual:
    def __init__(self, method):
        self.tree = tree(max_depth_individual, min_depth_individual, index_individual, method)
        self.max_depth = max_depth_individual
        self.min_depth = min_depth_individual
        self.index = index_individual
        self.fitness = None

    def fitness_individual(self, values):
        try:
            self.fitness = fitness_calculator(self, values)
        except Exception as e:
            logging.exception(e)

    def mutation_individual(self):
        alfa = random()
        if alfa < beta_mutation:
            depth = rand_index(1, self.max_depth)
            node_mutate = self.tree.subtree(depth)
            node_mutate.tree_mutation(self.index, self.max_depth)

    def crossover_individual(self, parent):
        alfa = random()
        if alfa <= alfa_crossover:
            depth = rand_index(1, self.max_depth)
            first_parent = deepcopy(self)
            snd_parent = deepcopy(parent)

            a = first_parent.tree.subtree(depth)
            b = snd_parent.tree.subtree(depth)

            Node.swap_subtree(a, b)
            return a, b
        else:
            return None, None

    def print_individual(self):
        return "Tree - {}".format(self.print_tree(0), end="\nFitness - {}".format(str(self.fitness)))


# Begin of the actual algorithm
def tree(depth, method):
    if depth == depth_max:
        return node(variables_size, depth)
    else:
        type_nodo = randint(0, 1)
        if type_nodo == 0 and depth_min <= depth < depth_max and method == "full":
            return node(variables_size, depth)
        else:
            operation = randint(1, functions_size)
            left = tree(depth_max, depth_min, variables_size, depth + 1)
            right = tree(depth_max, depth_min, variables_size, depth + 1)
            format_tree = Node(operator, operation, depth, left, right)
            return format_tree


def node(index, depth):  # Return a new node, who will be const or var
    type_nodo = randint(0, 1)
    if type_nodo == 0:
        return Node(variable, randint(1, index), depth)
    else:
        return Node(constant, random(), depth)


def get_operator(value):
    for k, v in functions.items():
        if v == value:
            return k


def rand_index(a, b):
    if b < a:
        return randint(b, a)
    return randint(a, b)


def fitness_calculator(individual, values):
    abs_error = 0.0
    index = len(values[0])
    for v in values:
        fitness_individual = individual.tree.evaluation(values[:index - 1])
        # The last index in the list is the f(x) himself
        expected_fitness = values[index - 1]
        abs_error += np.abs(fitness_individual - expected_fitness)
        return abs_error


def read_data():
    data = []
    with open(sys.argv[1], "rb") as archive:
        lines = archive.readlines()
        for line in lines:
            values = line.split()
            for v in values:
                values.append(float(v))
            data.append(values)
    return data


def create_population():
    individuals = []
    for _ in range(0, (population_size/2)):
        new_individual = Individual()
        individuals.append(new_individual)
    # Population size needs to always be a pair number
    for _ in range(0, (population_size/2)):
        new_individual = Individual(method="full")
        individuals.append(new_individual)
    return individuals


def do_tournament(population):
    random_individuals = []
    tournament = []
    n = 0
    while n < tournament_size:
        rand_ind = rand_index(0, population_size - 1)
        if rand_ind not in random_individuals:
            random_individuals.append(rand_ind)
            n += 1
    for rand in random_individuals:
        tournament.append(population[rand])

    # Sorting the tournament list to get the best individual
    tournament.sort(key=lambda k: k.fitness)
    return tournament[0]


def statistics(population):
    population_s = sorted(population, key=lambda k: k.fitness)
    global best_individual, best_fitness, worst_fitness, average_fitness, best_evolution, average_evolution, \
        worst_evolution, best_children, crossover_count, best_children_gen, best_individuals_crossover

    best_fitness = population_s[0].fitness
    best_individual = population_s[0]
    worst_fitness = population_s[-1].fitness
    average_fitness = np.average([k.fitness for k in population_s])
    if crossover_count > 0.0:
        best_individuals_crossover = (best_children / (crossover_count * 2.0)) * 100

    best_evolution.append(best_fitness)
    average_evolution.append(average_fitness)
    worst_evolution.append(worst_fitness)
    best_children_gen.append(best_individuals_crossover)


def run():
    global crossover_count, best_children, last_generation, number_data
    best_children = 0.0
    last_generation = 0.0
    n = 1

    data = read_data()
    number_data = len(data)

    # Write Output
    with open(sys.argv[2], "w") as archive:
        archive.write("Variáveis analisadas: " + str(number_data) + "\n")
        archive.write("Tamanho da População: " + str(population_size) + "\n")
        archive.write("Probabilidade - Crossover: " + str(alfa_crossover) + "\n")
        archive.write("Probabilidade - Mutação: " + str(beta_mutation) + "\n")
        archive.write("Tamanho - Torneio: " + str(tournament_size) + "\n")
        archive.write("Geração > Melhor Fitness > Fitness Médio > Melhor Indivíduo")

    population = create_population()
    for each in population:
        each.fitness_individual(data)

    statistics(population)

    if best_fitness < optimal_fitness:
        return

    while n < generations:
        new_population = []
        last_best = deepcopy(best_individual)

        for _ in range(0, (population_size/2)):
            a = do_tournament(population)
            b = do_tournament(population)

            a_child, b_child = a.crossover_individual(b)
            if a_child is not None and b_child is not None:
                crossover_count += 1
                fitness_calculator([a_child, b_child], data)
                descent = [a_child, b_child, a, b]
                descent.sort(key=lambda k: k.fitness)
                if descent[0] == (a_child or b_child):
                    best_children += 1  # I'm generating better children after crossbreed
                if descent[1] == (a_child or b_child):
                    best_children += 1
                if descent[0] or descent[1] not in new_population:
                    new_population.append(descent[0] or descent[1])

        for each in new_population:
            each.mutation_individual()

        new_population.append(last_best)
        # Using elitism to put the last best individual in the new population

        # Complete the population
        new_population.extend(create_population(population_size - len(new_population)))
        evolutionary_chart()


def evolutionary_chart():
    evolution_fig, evolution_ax = plot.subplots(4)
    evolution_fig.subplots_adjust(hspace=0.2)

    y1 = np.array(worst_evolution)
    y2 = np.array(average_evolution)
    y3 = np.array(best_evolution)
    y4 = np.array(best_children_gen)

    x1 = np.array(range(0, len(worst_evolution)))
    x2 = np.array(range(0, len(average_evolution)))
    x3 = np.array(range(0, len(best_evolution)))
    x4 = np.array(range(0, len(best_children_gen)))

    plot.setp(evolution_ax[0].get_xticklabels(), visible=False)
    plot.setp(evolution_ax[1].get_xticklabels(), visible=False)
    plot.setp(evolution_ax[2].get_xticklabels(), visible=False)
    plot.setp(evolution_ax[1].get_yticklabels(), visible=False)
    plot.setp(evolution_ax[2].get_yticklabels(), visible=False)

    evolution_ax[0].grid(True, which='both')
    evolution_ax[1].grid(True, which='both')
    evolution_ax[2].grid(True, which='both')
    evolution_ax[3].grid(True, which='both')

    evolution_ax[0].scatter(x4, y4, color='r', s=5.0)
    evolution_ax[1].scatter(x1, y1, color='y', s=5.0)
    evolution_ax[2].scatter(x2, y2, color='b', s=5.0)
    evolution_ax[3].scatter(x3, y3, color='g', s=5.0)

    evolution_fig.savefig(sys.argv[2] + "_evolution.png")


def clear_all_statistics():
    global best_evolution, worst_evolution, average_evolution, best_children_gen
    best_evolution = []
    worst_evolution = []
    average_evolution = []
    best_children_gen = []


def result(lb, la, lw, lc):
    best = []
    average = []
    worst = []
    children = []

    ind_best = [[row[r] for row in lb] for r in range(0, generations)]
    ind_ave = [[row[r] for row in la] for r in range(0, generations)]
    ind_wor = [[row[r] for row in lw] for r in range(0, generations)]
    ind_chl = [[row[r] for row in lc] for r in range(0, generations)]

    for i in range(0, generations):
        best.append(sum(ind_best[i])/executions)
        average.append(sum(ind_ave[i])/executions)
        worst.append(sum(ind_wor[i])/executions)
        children.append(sum(ind_chl[i])/executions)

        # Write Output
    with open(sys.argv[2] + "_result.txt", "w") as archive:
        archive.write("Variáveis analisadas: " + str(number_data) + "\n")
        archive.write("Tamanho da População: " + str(population_size) + "\n")
        archive.write("Probabilidade - Crossover: " + str(alfa_crossover) + "\n")
        archive.write("Probabilidade - Mutação: " + str(beta_mutation) + "\n")
        archive.write("Tamanho - Torneio: " + str(tournament_size) + "\n")
        archive.write("Média Aritmética das Execuções \n")
        archive.write("Geração > Melhor Fitness > Fitness Médio > Melhor Novo Indivíduo \n")
        for i in range(0, generations):
            archive.write("\n" + str(i + 1) + " > " + str(best[i]) + " > " + str(average[i]) +
                          " > " + str(worst[i]) + " > " + str(children[i]))

    evolutionary_chart(best, average, worst, children)


if __name__ == '__main__':
    best_all = []
    average_all = []
    worst_all = []
    children_proportion = []

    for _ in range(0, executions):
        run()
        best_all.append(best_evolution)
        average_all.append(average_evolution)
        worst_all.append(worst_evolution)
        children_proportion.append(best_children_gen)
        clear_all_statistics()

    result(best_all, average_all, worst_all, children_proportion)