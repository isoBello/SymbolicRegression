#!/usr/bin/env python3
from SymbolicRegression import PG
import numpy as np
import matplotlib.pyplot as plot

pop = []
best_evolution = []
average_evolution = []
worst_evolution = []
diversity = []
children_best_evolution = []
children_worst_evolution = []


def statistics():
    global pop, best_evolution, average_evolution, worst_evolution, diversity
    population_s = sorted(pop, key=lambda k: k.fitness)

    best_fitness = population_s[0].fitness
    worst_fitness = population_s[-1].fitness
    average_fitness = np.average([k.fitness for k in population_s])

    if PG.crossover_count > 0.0:
        proportion_best_children = (PG.number_best_children / (PG.crossover_count * 2.0)) * 100
        children_best_evolution.append(proportion_best_children)
    if PG.crossover_count > 0.0:
        proportion_worst_children = (PG.number_worst_children / (PG.crossover_count * 2.0)) * 100
        children_worst_evolution.append(proportion_worst_children)
    diversity_rate = (PG.new_ind / PG.generations) * 100

    best_evolution.append(best_fitness)
    average_evolution.append(average_fitness)
    worst_evolution.append(worst_fitness)
    diversity.append(diversity_rate)


def result_fitness(lb, la, lw):
    best = []
    average = []
    worst = []
    i = 0

    temp_best = [[row[i] for row in lb] for i in range(0, PG.generations)]
    temp_ave = [[row[i] for row in la] for i in range(0, PG.generations)]
    temp_worst = [[row[i] for row in lw] for i in range(0, PG.generations)]

    for gen in range(0, PG.generations):
        best.append(sum(temp_best[gen]) / PG.executions)
        average.append(sum(temp_ave[gen]) / PG.executions)
        worst.append(sum(temp_worst[gen]) / PG.executions)

        # Write Output
    with open("Tests/circle_fitness" + "_result.txt", "w") as archive:
        archive.write("Variáveis analisadas: " + str(PG.number_data) + "\n")
        archive.write("Tamanho da População: " + str(PG.population_size) + "\n")
        archive.write("Probabilidade - Crossover: " + str(PG.alfa_crossover) + "\n")
        archive.write("Probabilidade - Mutação: " + str(PG.beta_mutation) + "\n")
        archive.write("Tamanho - Torneio: " + str(PG.tournament_size) + "\n")
        archive.write("Média das execuções" + "\n")
        archive.write("Geração > Melhor Fitness > Fitness Médio > Pior Fitness")
        for ind in range(0, PG.generations):
            archive.write("\n" + str(ind + 1) + " > " + str(best[ind]) + " > " + str(average[ind]) +
                          " > " + str(worst[ind]))

    evolutionary_fitness_chart(best, average, worst, None, False)


def result_individuals(lcb, lcw, ld):
    best = []
    worst = []
    div = []

    temp_best = [[row[i] for row in lcb] for i in range(0, PG.generations)]
    temp_worst = [[row[i] for row in lcw] for i in range(0, PG.generations)]
    temp_div = [[row[i] for row in ld] for i in range(0, PG.generations)]

    for gen in range(0, PG.generations):
        best.append(sum(temp_best[gen]) / PG.executions)
        worst.append(sum(temp_worst[gen]) / PG.executions)
        div.append(sum(temp_div[gen]) / PG.executions)

        # Write Output
    with open("Tests/circle_individuals" + "_result.txt", "w") as archive:
        archive.write("Variáveis analisadas: " + str(PG.number_data) + "\n")
        archive.write("Tamanho da População: " + str(PG.population_size) + "\n")
        archive.write("Probabilidade - Crossover: " + str(PG.alfa_crossover) + "\n")
        archive.write("Probabilidade - Mutação: " + str(PG.beta_mutation) + "\n")
        archive.write("Tamanho - Torneio: " + str(PG.tournament_size) + "\n")
        archive.write("Média das execuções" + "\n")
        archive.write("Geração > Proporção de Filhos Melhores > Proporção de Filhos Piores > Indivíduos Repetidos")
        for ind in range(0, PG.generations):
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
    global best_evolution, worst_evolution, average_evolution, diversity
    best_evolution = []
    worst_evolution = []
    average_evolution = []
    diversity = []
    PG.population = []
