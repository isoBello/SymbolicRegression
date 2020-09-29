#!/usr/bin/env python
# -*- coding: UTF-8 -*-
from SymbolicRegression import PG
from SymbolicRegression import Statistics

if __name__ == '__main__':
    best_all = []
    average_all = []
    worst_all = []

    children_best = []
    children_worst = []
    diversity_all = []

    bests = {}

    for i in range(0, PG.executions):
        PG.run(i)
        menor_fitness = 999999

        # Fitness Graphic
        best_all.append(Statistics.best_evolution)
        average_all.append(Statistics.average_evolution)
        worst_all.append(Statistics.worst_evolution)

        # Individuals Graphic
        children_best.append(Statistics.children_best_evolution)
        children_worst.append(Statistics.children_worst_evolution)
        diversity_all.append(Statistics.diversity)

        # Write bests trees
        for ind in Statistics.best_evolution:
            if ind < menor_fitness:
                menor_fitness = ind

        for pop_tree in PG.population:
            if pop_tree.fitness == menor_fitness:
                bests[i] = pop_tree.print_individual()

        Statistics.clear_all_statistics()
        children_best_evolution = []
        children_worst_evolution = []
        population = []

    # Write Outputs
    Statistics.result_fitness(best_all, average_all, worst_all)
    Statistics.result_individuals(children_best, children_worst, diversity_all)
    Statistics.result_trees(bests)
    print("finished - look at results :)")
