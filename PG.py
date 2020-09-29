#!/usr/bin/env python3
from copy import deepcopy
from random import randint
from collections import OrderedDict
from SymbolicRegression import Individual
from SymbolicRegression import Tree
from SymbolicRegression import Statistics

# Constants used to the experimental analysis of the PG
executions = 30
depth_individual = 5
alfa_crossover = 0.6
beta_mutation = 0.2
tournament_size = 7
population_size = 20
generations = 3550
write_count = 0

# Variables used to present the results of the PG
best_individual = None
best_fitness = 0.0
worst_fitness = 0.0
average_fitness = 0.0

population = []

crossover_count = 0
number_best_children = 0
number_worst_children = 0
proportion_best_children = 0
proportion_worst_children = 0
new_ind = 0
number_data = 1


# Begin of the actual algorithm
def read_data():
    data = {}
    temp = []

    with open("datasets-TP1/SR_circle.txt", "r") as archive:
        lines = archive.readlines()
        for line in lines:
            values = line.split()
            for v in values[:-1]:
                v = float(v)
                temp.append(v)

            data[float(values[-1])] = temp
            temp = []

    # Getting the max number of variables we can have in our PG
    Tree.variables_size = max(len(v) for k, v in data.items())

    # Getting the range of our constants, based in our dataset
    Tree.interval_max = max(max(v) for k, v in data.items())
    Tree.interval_min = min(min(v) for k, v in data.items())

    return data


def create_population(size):
    # Method who creates the population
    individuals = []
    for k in range(0, int(size / 2)):
        new_individual = Individual.Individual(k, False)
        individuals.append(new_individual)

    # Population size needs to always be a pair number
    for j in range(int(size / 2), 0):
        new_individual = Individual.Individual(j, True)
        individuals.append(new_individual)
    return individuals


def tournament(pop):
    # Tournament between the population
    random_individuals = []
    n = 0
    while n < tournament_size:
        rand_ind = randint(0, (len(pop) - 1))
        if rand_ind not in random_individuals:
            random_individuals.append(rand_ind)
            n += 1

    chosen = [pop[value] for value in random_individuals if pop[value].fitness is not None]

    # Sorting the tournament list to get the best individual

    chosen.sort(key=lambda k: k.fitness)
    return chosen[0]


def run(exe):
    global write_count, crossover_count, number_best_children, number_worst_children, number_data, new_ind, population, \
        best_fitness, worst_fitness, average_fitness

    new_ind = 0
    number_best_children = 0
    number_worst_children = 0

    n = 1
    string_arquivo = ""

    data = read_data()
    number_data = len(data)

    population = create_population(population_size)
    for each in population:
        if each is not None:
            each.fit(data)

    Statistics.pop = [v for v in population if v.fitness is not None]
    Statistics.statistics()

    while n < generations:

        new_population = []
        last_best = deepcopy(best_individual)

        bchild = 0
        wchild = 0
        new_ind = 0

        for j in range(0, int(population_size / 2)):
            a = tournament(population)
            b = tournament(population)

            c, d = a.crossover(b, data)
            if c is not None and d is not None:
                crossover_count += 1
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
                    new_population.append(descent[0])   # Putting this individuals in our population
                if descent[1] not in new_population:
                    new_population.append(descent[1])   # Putting this individuals in our population
                if descent[0] not in population:
                    new_ind += 1
                if descent[1] not in population:
                    new_ind += 1

        best_fitness = min(each.fitness for each in population if each.fitness is not None)
        worst_fitness = max(each.fitness for each in population if each.fitness is not None)
        average_fitness = sum(each.fitness for each in population if each.fitness is not None)/len(population)

        string_arquivo += ("\n" + str(n) + " > " + str(best_fitness) + " > " + str(average_fitness) +
                           " > " + str(worst_fitness))

        for each in new_population:
            each.mutation()

        new_population.append(last_best)
        # Using elitism to put the last best individual in the new population

        # Complete the population
        n_ind = population_size - len(new_population)

        string_arquivo += " > " + str(bchild) + " > " + str(wchild) + " > " + str(new_ind)

        restant = create_population(n_ind)
        new_population.extend(restant)

        # Calculate the fitness of the new population
        for each in new_population:
            if each is not None:
                each.fit(data)
            else:
                new_population.remove(each)
                new_population.append(Individual.Individual(population_size+1))

        Statistics.pop = [v for v in new_population if v.fitness is not None]
        Statistics.statistics()

        population = new_population
        n += 1

        with open("Tests/circle" + str(exe) + ".txt", "w") as archive:
            archive.write("Variáveis analisadas: " + str(number_data) + "\n")
            archive.write("Tamanho da População: " + str(population_size) + "\n")
            archive.write("Probabilidade - Crossover: " + str(alfa_crossover) + "\n")
            archive.write("Probabilidade - Mutação: " + str(beta_mutation) + "\n")
            archive.write("Tamanho - Torneio: " + str(tournament_size) + "\n")
            archive.write("Geração > Melhor Fitness > Fitness Médio > Pior Fitness > "
                          "Número de Melhores Filhos > Número de Piores Filhos > Indivíduos Repetidos")
            archive.write("\n".join(list(OrderedDict.fromkeys(string_arquivo.split("\n")))))

    Statistics.evolutionary_fitness_chart(Statistics.best_evolution, Statistics.average_evolution,
                                          Statistics.worst_evolution, exe, True)
