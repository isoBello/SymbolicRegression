#!/usr/bin/env python3
import logging
from SymbolicRegression import PG
from SymbolicRegression import Tree
from copy import deepcopy
from random import randint, uniform


# This class represents every individual in our population
class Individual:
    def __init__(self, id, method=False, tree=None):
        if tree is not None:
            self.tree = tree
        else:
            if not method:
                self.tree = Tree.tree(0, method, dmax=5)
            else:
                self.tree = Tree.tree(0, method, dmax=Tree.depth_max)
        self.id = id
        self.fitness = None

    def fit(self, values):
        try:
            self.fitness = Tree.fitness_calculator(self, values)
        except Exception as e:
            logging.exception(e)

    def print_individual(self):
        string_individual = "Tree: \n" + Tree.print_tree(self.tree) + "\nFitness " + str(self.fitness)
        return string_individual

    def mutation(self):
        alfa = uniform(0, 1)
        if alfa < PG.beta_mutation:
            depth = randint(1, Tree.depth_max)
            node_mutate = self.tree.subtree(depth)
            node_mutate.tree_mutation(Tree.depth_max)

    def crossover(self, parent, data):
        alfa = uniform(0, 1)
        if alfa <= PG.alfa_crossover:
            depth = randint(1, Tree.depth_max)
            a = deepcopy(self)
            b = deepcopy(parent)
            c = a.tree.subtree(depth)
            d = b.tree.subtree(depth)

            c.swap_subtree(d)
            c_ind = Individual(a.id+1, tree=c)
            d_ind = Individual(b.id+1, tree=d)

            for child in [c_ind, d_ind]:
                child.fit(data)

            return c_ind, d_ind
        else:
            return None, None
