#!/usr/bin/env python3
from copy import deepcopy
from random import randint, random
import math
import logging
import numpy as np
import matplotlib.pyplot as plot

# import sys

# Constants used to the experimental analysis of the PG
executions = 10
depth_individual = 7
alfa_crossover = 0.9
beta_mutation = 0.05
tournament_size = 2
population_size = 10
elite_individuals = 2
generations = 20
optimal_fitness = 0.0

# Variables used to present the results of the PG
best_individual = None
best_fitness = 0.0
worst_fitness = 0.0
average_fitness = 0.0

best_evolution = []
average_evolution = []
worst_evolution = []
children_evolution = []

last_generation = 0.0
crossover_count = 0
number_best_children = 0
proportion_best_children = 0
number_data = 1

# Tree characteristics
operator = "Operator"
constant = "Constant"
variable = "Variable"
functions = {1: 'add', 2: 'sub', 3: 'mul', 4: 'div', 5: 'sen', 6: 'cos'}
functions_size = len(functions)
variables = {'x': 1}
variables_size = len(variables)
interval_constants = range(-10, 10)
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
            return values[self.value - 1]

    @staticmethod
    def function_evaluation(opration, a, b):
        if opration == 1:
            return a + b
        elif opration == 2:
            return a - b
        elif opration == 3:
            return a * b
        elif opration == 4:
            return a / b if b else 0  # Prevents errors from division: num / 0
        elif opration == 5:
            return math.sin(a)
        elif opration == 6:
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

    def swap_subtree(self, b):  # Swap between two nodes
        self.__dict__, b.__dict__ = b.__dict__, self.__dict__

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
                new_tree = tree(self.height, '')
                Node.swap_subtree(self, new_tree)

    def print_tree(self, level):
        str_tree = '  ' * level
        if self.type == variable:
            str_tree += "x" + str(self.value)
        elif self.type == operator:
            str_tree += str(get_operator(self.value))
        else:
            str_tree += str(f'{self.value:.4g}')

        if self.leaf_left is not None:
            str_tree += self.leaf_left.print_tree(level + 1)
        if self.leaf_right is not None:
            str_tree += self.leaf_right.print_tree(level + 1)

        return str_tree


class Individual:
    def __init__(self, method):
        self.tree = tree(max_depth_individual, method)
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
            a_ind = Individual(a)
            b_ind = Individual(b)
            return a_ind, b_ind
        else:
            return None, None

    def print_individual(self):
        string_individual = "Tree " + self.tree.print_tree(0) + "\nFitness " + str(self.fitness)
        return string_individual


# Begin of the actual algorithm
def tree(depth, method):
    # Method defines if it will be a full or grow tree. If True = Full, otherwise = Grow. Based on ramped-half-and-half
    try:
        if depth == depth_max:
            return node(depth)
        else:
            type_nodo = rand_index(0, 1)
            if type_nodo == 0 and depth_min <= depth and not method:
                return node(depth)
            else:
                operation = rand_index(1, functions_size)
                left = tree(depth + 1, method)
                right = tree(depth + 1, method)
                format_tree = Node(operator, operation, depth, left, right)
        return format_tree
    except Exception as e:
        logging.exception(e)


def node(depth):  # Return a new node, who will be const or var
    no = randint(0, 1)
    if no == 0:
        return Node(variable, rand_index(1, variables_size), depth)
    else:
        return Node(constant, random(), depth)


def get_operator(key):
    for k, v in functions.items():
        if k == key:
            return v


def rand_index(a, b):
    if b < a:
        return randint(b, a)
    return randint(a, b)


def fitness_calculator(individual, values):
    abs_error = 0.0
    index = len(values[0])
    for v in values:
        fit = individual.tree.evaluation(v[:index - 1])
        # The last index in the list is the f(x) himself
        expected_fitness = v[index - 1]
        abs_error += np.abs(fit - expected_fitness)
        return abs_error


def read_data():
    data = []
    temp = []
    with open("datasets-TP1/SR_circle.txt", "r") as archive:
        lines = archive.readlines()
        for line in lines:
            values = line.split()
            for v in values:
                v = float(v)
                temp.append(v)
            data.append(temp)
    return data


def create_population(size):
    individuals = []
    for _ in range(0, int(size / 2)):
        new_individual = Individual(method='')
        individuals.append(new_individual)
    # Population size needs to always be a pair number
    for _ in range(0, int(size / 2)):
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
        worst_evolution, children_evolution, crossover_count, number_best_children, proportion_best_children

    best_fitness = population_s[0].fitness
    best_individual = population_s[0]
    worst_fitness = population_s[-1].fitness
    average_fitness = np.average([k.fitness for k in population_s])
    if crossover_count > 0.0:
        proportion_best_children = (number_best_children / (crossover_count * 2.0)) * 100

    best_evolution.append(best_fitness)
    average_evolution.append(average_fitness)
    worst_evolution.append(worst_fitness)
    children_evolution.append(proportion_best_children)


def run():
    global crossover_count, number_best_children, last_generation, number_data
    number_best_children = 0.0
    last_generation = 0.0
    n = 1

    data = read_data()
    number_data = len(data)

    write_output()

    population = create_population(population_size)
    for each in population:
        each.fitness_individual(data)

    statistics(population)

    if best_fitness < optimal_fitness:
        return

    while n < generations:
        new_population = []
        last_best = deepcopy(best_individual)

        for _ in range(0, int(population_size / 2)):
            a = do_tournament(population)
            b = do_tournament(population)

            c, d = a.crossover_individual(b)
            if c is not None and d is not None:
                crossover_count += 1
                for child in [c, d]:
                    child.fitness_individual(data)
                descent = [c, d, a, b]
                descent.sort(key=lambda k: k.fitness)
                if descent[0] == (c or d):
                    number_best_children += 1  # I'm generating better children after crossbreed
                if descent[1] == (c or d):
                    number_best_children += 1
                if descent[0] or descent[1] not in new_population:
                    new_population.append(descent[0] or descent[1])

        for each in new_population:
            each.mutation_individual()

        new_population.append(last_best)
        # Using elitism to put the last best individual in the new population

        # Complete the population
        population_r = create_population(population_size - len(new_population))
        new_population.extend(population_r)
        evolutionary_chart(best_evolution, average_evolution, worst_evolution, children_evolution)


def write_output():
    # Write Output
    with open("tests/circle.txt", "w") as archive:
        archive.write("Variáveis analisadas: " + str(number_data) + "\n")
        archive.write("Tamanho da População: " + str(population_size) + "\n")
        archive.write("Probabilidade - Crossover: " + str(alfa_crossover) + "\n")
        archive.write("Probabilidade - Mutação: " + str(beta_mutation) + "\n")
        archive.write("Tamanho - Torneio: " + str(tournament_size) + "\n")
        archive.write("Geração > Melhor Fitness > Fitness Médio > Melhor Indivíduo")


def evolutionary_chart(lb, la, lw, lc):
    evolution_fig, evolution_ax = plot.subplots(4)
    evolution_fig.subplots_adjust(hspace=0.2)

    y1 = np.array(lw)
    y2 = np.array(la)
    y3 = np.array(lb)
    y4 = np.array(lc)

    x1 = np.array(range(0, len(lw)))
    x2 = np.array(range(0, len(la)))
    x3 = np.array(range(0, len(lb)))
    x4 = np.array(range(0, len(lc)))

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

    evolution_fig.savefig("tests/circle" + "_evolution.png")
    plot.close(evolution_fig)


def clear_all_statistics():
    global best_evolution, worst_evolution, average_evolution, number_best_children
    best_evolution = []
    worst_evolution = []
    average_evolution = []
    number_best_children = []


def result(lb, la, lw, lc):
    best = []
    average = []
    worst = []
    children = []

    ind_best = [[row[r] for row in lb] for r in range(0, generations)]
    ind_ave = [[row[r] for row in la] for r in range(0, generations)]
    ind_wor = [[row[r] for row in lw] for r in range(0, generations)]
    ind_chl = [[row[r] for row in lc] for r in range(0, generations)]

    for gen in range(0, generations):
        best.append(sum(ind_best[gen]) / executions)
        average.append(sum(ind_ave[gen]) / executions)
        worst.append(sum(ind_wor[gen]) / executions)
        children.append(sum(ind_chl[gen]) / executions)

        # Write Output
    with open("tests/circle" + "_result.txt", "w") as archive:
        archive.write("Variáveis analisadas: " + str(number_data) + "\n")
        archive.write("Tamanho da População: " + str(population_size) + "\n")
        archive.write("Probabilidade - Crossover: " + str(alfa_crossover) + "\n")
        archive.write("Probabilidade - Mutação: " + str(beta_mutation) + "\n")
        archive.write("Tamanho - Torneio: " + str(tournament_size) + "\n")
        archive.write("Média Aritmética das Execuções \n")
        archive.write("Geração > Melhor Fitness > Fitness Médio > Melhor Novo Indivíduo \n")
        for ind in range(0, generations):
            archive.write("\n" + str(ind + 1) + " > " + str(best[ind]) + " > " + str(average[ind]) +
                          " > " + str(worst[ind]) + " > " + str(children[ind]))

    evolutionary_chart(best, average, worst, children)


def main():
    best_all = []
    average_all = []
    worst_all = []
    children_proportion = []

    for i in range(0, executions):
        run()
        best_all.append(best_evolution)
        average_all.append(average_evolution)
        worst_all.append(worst_evolution)
        children_proportion.append(number_best_children)
        clear_all_statistics()

    result(best_all, average_all, worst_all, children_proportion)


if __name__ == '__main__':
    main()
    print("finished - look at results :)")
