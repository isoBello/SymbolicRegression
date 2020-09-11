#!/usr/bin/env python3
from copy import deepcopy
from random import randint, random
import math
import logging
import numpy as np
import matplotlib.pyplot as plot
from collections import OrderedDict

# Constants used to the experimental analysis of the PG
executions = 30
depth_individual = 7
alfa_crossover = 0.6
beta_mutation = 0.3
tournament_size = 2
population_size = 3500
elite_individuals = 2
generations = 100
optimal_fitness = 0.0
write_count = 0

# Variables used to present the results of the PG
best_individual = None
best_fitness = 0.0
worst_fitness = 0.0
average_fitness = 0.0

population = []
best_evolution = []
average_evolution = []
worst_evolution = []

children_best_evolution = []
children_worst_evolution = []
diversity = []

last_generation = 0.0
crossover_count = 0
number_best_children = 0
number_worst_children = 0
proportion_best_children = 0
proportion_worst_children = 0
same_individuals = 0
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
depth_max = 7
depth_min = 2

# Individual characteristics
min_depth_individual = 1
max_depth_individual = 2

# Folder Tests
folder = str("tests_pop_" + str(population_size) + "_gen_" + str(generations))


class Node:
    def __init__(self, no, value, height, leaf_left=None, leaf_right=None):
        self.type = no
        self.value = value
        self.height = height
        self.leaf_left = leaf_left
        if no == operator and 5 <= value <= 6:
            self.leaf_right = Node(constant, 0.0, height + 1)
            # Used in right leaves with sen or cos values, 'cause they need a constant after showed
            # (i.e we don't have sen + 2, just sen 2)
        else:
            self.leaf_right = leaf_right

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
                elif self.leaf_right is not None:
                    return self.leaf_right.subtree(depth)
                else:
                    return self
            else:
                if self.leaf_right is not None:
                    return self.leaf_right.subtree(depth)
                elif self.leaf_left is not None:
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

    def print_tree(self, level=0):
        str_tree = '  ' * level

        if level == 0:
            str_tree += "root: "
        else:
            str_tree.replace("root: ", "")

        if self.type == variable:
            str_tree += "x" + str(self.value)
        elif self.type == operator:
            str_tree += str(get_operator(self.value))
        else:
            str_tree += str(f'{self.value:.4g}')

        str_tree += "\n"

        if self.leaf_left is not None:
            str_tree += " Nó esquerdo - Altura " + str(self.height) + ": " + self.leaf_left.print_tree(level + 1)
        if self.leaf_right is not None:
            str_tree += " Nó direito - Altura " + str(self.height) + ":" + self.leaf_right.print_tree(level + 1)

        return str_tree


class Individual:
    def __init__(self, id, method=False):
        self.tree = tree(max_depth_individual, method)
        self.id = id
        self.fitness = None

    def fitness_individual(self, values):
        try:
            self.fitness = fitness_calculator(self, values)
        except Exception as e:
            logging.exception(e)

    def mutation_individual(self):
        alfa = random()
        if alfa < beta_mutation:
            depth = rand_index(1, depth_max)
            node_mutate = self.tree.subtree(depth)
            node_mutate.tree_mutation(self.id, depth_max)

    def crossover_individual(self, parent):
        alfa = random()
        if alfa <= alfa_crossover:
            depth = rand_index(1, depth_max)
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
        string_individual = "Tree: \n" + self.tree.print_tree() + "\nFitness " + str(self.fitness)
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


def create_population(size):
    individuals = []
    for i in range(0, int(size / 2)):
        new_individual = Individual(i, False)
        individuals.append(new_individual)

    # Population size needs to always be a pair number
    for j in range(0, int(size / 2)):
        new_individual = Individual(j, True)
        individuals.append(new_individual)
    return individuals


def do_tournament(pop):
    random_individuals = []
    n = 0
    while n < tournament_size:
        rand_ind = randint(0, (len(pop) - 1))
        if rand_ind not in random_individuals:
            random_individuals.append(rand_ind)
            n += 1
    tournament = [pop[value] for value in random_individuals]

    # Sorting the tournament list to get the best individual
    tournament.sort(key=lambda k: k.fitness)
    return tournament[0]


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


def run(exe):
    global write_count, crossover_count, number_best_children, number_worst_children, last_generation, \
        number_data, same_individuals, population

    last_generation = 0.0
    same_individuals = 0
    number_best_children = 0
    number_worst_children = 0

    n = 1
    string_arquivo = ""

    data = read_data()
    number_data = len(data)

    population = create_population(population_size)
    for each in population:
        each.fitness_individual(data)

    statistics(population)

    while n < generations:

        new_population = []
        last_best = deepcopy(best_individual)

        bchild = 0
        wchild = 0

        for j in range(0, int(population_size / 2)):
            a = do_tournament(population)
            b = do_tournament(population)

            c, d = a.crossover_individual(b)
            if c is not None and d is not None:
                crossover_count += 1
                for child in [c, d]:
                    child.fitness_individual(data)
                descent = [c, d, a, b]
                descent.sort(key=lambda key: key.fitness)

                if descent[0] == (c or d):
                    number_best_children += 1  # I'm generating better children after crossbreed
                    bchild += 1
                if descent[1] == (c or d):
                    number_best_children += 1  # I'm generating better children after crossbreed
                    bchild += 1

                fit_ave = (a.fitness + b.fitness) / 2
                if c.fitness > fit_ave:
                    number_worst_children += 1  # I'm generating worst children after crossbreed
                    wchild += 1
                if d.fitness > fit_ave:
                    number_worst_children += 1  # I'm generating worst children after crossbreed
                    wchild += 1

                if descent[0] not in new_population:
                    new_population.append(descent[0])
                if descent[1] not in new_population:
                    new_population.append(descent[1])

        string_arquivo += ("\n" + str(n) + " > " + str(best_fitness) + " > " + str(average_fitness) +
                           " > " + str(worst_fitness))

        for each in new_population:
            each.mutation_individual()

        new_population.append(last_best)
        # Using elitism to put the last best individual in the new population

        if best_fitness <= optimal_fitness:
            for pop in population:
                if pop.fitness == best_fitness:
                    print("Find one looking good individual, check out: {}".format(pop.print_individual()), end="")
                    return

        # Complete the population
        same_individuals = population_size - len(new_population)

        string_arquivo += " > " + str(bchild) + " > " + str(wchild) + " > " + str(same_individuals)

        restant = create_population(same_individuals)
        new_population.extend(restant)

        # Calculate the fitness of the new population
        for each in new_population:
            each.fitness_individual(data)

        statistics(new_population)

        last_generation = n

        population = new_population
        n += 1

        with open("Tests/circle.txt", "w") as archive:
            archive.write("Variáveis analisadas: " + str(number_data) + "\n")
            archive.write("Tamanho da População: " + str(population_size) + "\n")
            archive.write("Probabilidade - Crossover: " + str(alfa_crossover) + "\n")
            archive.write("Probabilidade - Mutação: " + str(beta_mutation) + "\n")
            archive.write("Tamanho - Torneio: " + str(tournament_size) + "\n")
            archive.write("Geração > Melhor Fitness > Fitness Médio > Pior Fitness > "
                          "Número de Melhores Filhos > Número de Piores Filhos > Indivíduos Repetidos")
            archive.write("\n".join(list(OrderedDict.fromkeys(string_arquivo.split("\n")))))

    evolutionary_fitness_chart(best_evolution, average_evolution, worst_evolution, exe, True)


def statistics(pop):
    population_s = sorted(pop, key=lambda k: k.fitness)
    global best_fitness, best_individual, worst_fitness, average_fitness, proportion_best_children, \
        best_evolution, average_evolution, worst_evolution, children_best_evolution, \
        children_worst_evolution, diversity, crossover_count, number_best_children, number_worst_children, \
        proportion_worst_children

    best_fitness = population_s[0].fitness
    best_individual = population_s[0]

    worst_fitness = population_s[-1].fitness
    average_fitness = np.average([k.fitness for k in population_s])

    if crossover_count > 0.0:
        proportion_best_children = (number_best_children / (crossover_count * 2.0)) * 100
    if crossover_count > 0.0:
        proportion_worst_children = (number_worst_children / (crossover_count * 2.0)) * 100
    diversity_rate = (same_individuals / generations) * 100

    best_evolution.append(best_fitness)
    average_evolution.append(average_fitness)
    worst_evolution.append(worst_fitness)
    children_best_evolution.append(proportion_best_children)
    children_worst_evolution.append(proportion_worst_children)
    diversity.append(diversity_rate)


def result_fitness(lb, la, lw):
    best = []
    average = []
    worst = []

    temp_best = [[row[i] for row in lb] for i in range(0, generations)]
    temp_ave = [[row[i] for row in la] for i in range(0, generations)]
    temp_worst = [[row[i] for row in lw] for i in range(0, generations)]

    for gen in range(0, generations):
        best.append(sum(temp_best[gen]) / executions)
        average.append(sum(temp_ave[gen]) / executions)
        worst.append(sum(temp_worst[gen]) / executions)

        # Write Output
    with open("Tests/circle_fitness" + "_result.txt", "w") as archive:
        archive.write("Variáveis analisadas: " + str(number_data) + "\n")
        archive.write("Tamanho da População: " + str(population_size) + "\n")
        archive.write("Probabilidade - Crossover: " + str(alfa_crossover) + "\n")
        archive.write("Probabilidade - Mutação: " + str(beta_mutation) + "\n")
        archive.write("Tamanho - Torneio: " + str(tournament_size) + "\n")
        archive.write("Média das execuções" + "\n")
        archive.write("Geração > Melhor Fitness > Fitness Médio > Pior Fitness")
        for ind in range(0, generations):
            archive.write("\n" + str(ind + 1) + " > " + str(best[ind]) + " > " + str(average[ind]) +
                          " > " + str(worst[ind]))

    evolutionary_fitness_chart(best, average, worst, None, False)


def result_individuals(lcb, lcw, ld):
    best = []
    worst = []
    div = []

    temp_best = [[row[i] for row in lcb] for i in range(0, generations)]
    temp_worst = [[row[i] for row in lcw] for i in range(0, generations)]
    temp_div = [[row[i] for row in ld] for i in range(0, generations)]

    for gen in range(0, generations):
        best.append(sum(temp_best[gen]) / executions)
        worst.append(sum(temp_worst[gen]) / executions)
        div.append(sum(temp_div[gen]) / executions)

        # Write Output
    with open("Tests/circle_individuals" + "_result.txt", "w") as archive:
        archive.write("Variáveis analisadas: " + str(number_data) + "\n")
        archive.write("Tamanho da População: " + str(population_size) + "\n")
        archive.write("Probabilidade - Crossover: " + str(alfa_crossover) + "\n")
        archive.write("Probabilidade - Mutação: " + str(beta_mutation) + "\n")
        archive.write("Tamanho - Torneio: " + str(tournament_size) + "\n")
        archive.write("Média das execuções" + "\n")
        archive.write("Geração > Proporção de Filhos Melhores > Proporção de Filhos Piores > Indivíduos Repetidos")
        for ind in range(0, generations):
            archive.write("\n" + str(ind + 1) + " > " + str(best[ind]) + " > " + str(worst[ind]) +
                          " > " + str(div[ind]))

    evolutionary_individuals_chart(best, worst, div)


def result_trees(best):
    with open("Tests/bests.txt", "w") as archive:
        archive.write("Melhores Indivíduos: " + "\n")
        for k, b in best.items():
            archive.write("\n Execução: " + str(k) + "\n" + b + "\n")


def evolutionary_fitness_chart(lb, la, lw, exe, check):
    evolution_fig, evolution_ax = plot.subplots()

    y1 = np.array(lb)
    y2 = np.array(lw)
    y3 = np.array(la)

    x1 = np.array(range(0, len(lb)))
    x2 = np.array(range(0, len(lw)))
    x3 = np.array(range(0, len(la)))

    lines = plot.plot(x1, y1, x2, y2, x3, y3)

    l1, l2, l3 = lines
    plot.setp(lines, linestyle="--")

    plot.setp(l1, linewidth=2, color='g')  # best
    plot.setp(l2, linewidth=1, color='r')  # worst
    plot.setp(l3, linewidth=1, color='b')  # average

    custom_lines = [l1, l2, l3]

    title = plot.title("Média da Fitness do PG")

    lines_legend = plot.legend(custom_lines, ['Melhor', 'Pior', 'Média'])
    plot.xlabel("Gerações")
    plot.ylabel("Fitness")

    plot.setp(title)
    plot.setp(lines_legend)

    plot.grid(b=True, which='major', color='#666666', linestyle='-')
    plot.minorticks_on()
    plot.grid(b=True, which='minor', color='#999999', linestyle='-', alpha=0.2)

    if check:
        evolution_fig.savefig("Tests/circle" + "_evolution_exec_" + str(exe) + ".png")
    else:
        evolution_fig.savefig("Tests/circle" + "_evolution.png")


def evolutionary_individuals_chart(lbc, lwc, ld):
    evolution_fig, evolution_ax = plot.subplots()

    y1 = np.array(lbc)
    y2 = np.array(lwc)
    y3 = np.array(ld)

    x1 = np.array(range(0, len(lbc)))
    x2 = np.array(range(0, len(lwc)))
    x3 = np.array(range(0, len(ld)))

    lines = plot.plot(x1, y1, x2, y2, x3, y3)

    l1, l2, l3 = lines
    plot.setp(lines, linestyle="--")

    plot.setp(l1, linewidth=2, color='g')  # best
    plot.setp(l2, linewidth=1, color='r')  # worst
    plot.setp(l3, linewidth=1, color='b')  # average

    custom_lines = [l1, l2, l3]

    title = plot.title("Proporção dos Indivíduos no PG")
    proportions = plot.legend(custom_lines, ['Filhos Melhores', 'Filhos Piores', 'Indivíduos Repetidos'])
    plot.xlabel("Gerações")
    plot.ylabel("Porporção")

    plot.setp(title)
    plot.setp(proportions)

    plot.grid(b=True, which='major', color='#666666', linestyle='-')
    plot.minorticks_on()
    plot.grid(b=True, which='minor', color='#999999', linestyle='-', alpha=0.2)

    evolution_fig.savefig("Tests/circle" + "_proportion.png")


def clear_all_statistics():
    global best_evolution, worst_evolution, average_evolution, children_best_evolution, \
        children_worst_evolution, diversity, population
    best_evolution = []
    worst_evolution = []
    average_evolution = []
    children_best_evolution = []
    children_worst_evolution = []
    diversity = []
    population = []


def main():
    best_all = []
    average_all = []
    worst_all = []

    children_best = []
    children_worst = []
    diversity_all = []

    bests = {}

    for i in range(0, executions):
        run(i)
        menor_fitness = 100

        # Fitness Graphic
        best_all.append(best_evolution)
        average_all.append(average_evolution)
        worst_all.append(worst_evolution)

        # Individuals Graphic
        children_best.append(children_best_evolution)
        children_worst.append(children_worst_evolution)
        diversity_all.append(diversity)

        # Write bests trees
        for ind in best_evolution:
            if ind < menor_fitness:
                menor_fitness = ind

        for pop_tree in population:
            if pop_tree.fitness == menor_fitness:
                bests[i] = pop_tree.print_individual()

        clear_all_statistics()

    # Write Outputs
    result_fitness(best_all, average_all, worst_all)
    result_individuals(children_best, children_worst, diversity_all)
    result_trees(bests)


if __name__ == '__main__':
    main()
    print("finished - look at results :)")
