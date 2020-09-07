#!/usr/bin/env python3
from copy import deepcopy
from random import randint, random
import math
import numpy as np
import matplotlib.pyplot as plot

# Constants used to the experimental analysis of the PG
executions = 10
depth_individual = 7
min_depth_individual = 1
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
do_crossover = 0.0
best_children_gen = []
proportion_best_children = []

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

    def mutation(self, max_depth, index):
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


def change_subtree(max_depth=depth_max, min_depth=depth_min, variables_range=variables_size, depth=0):
    if depth == max_depth:
        return node(variables_range, depth)
    else:
        type_nodo = randint(0, 1)
        if type_nodo == 0 and min_depth <= depth < max_depth:
            return node(variables_range, depth)
        else:
            operation = randint(1, functions_size)
            left = change_subtree(max_depth, min_depth, variables_size, depth + 1)
            right = change_subtree(max_depth, min_depth, variables_size, depth + 1)
            tree = Node(operator, operation, depth, left, right)
            return tree


def node(index, depth):  # Return a new node, who will be const or var
    type_nodo = randint(0, 1)
    if type_nodo == 0:
        return Node(variable, randint(1, index), depth)
    else:
        return Node(constant, random(), depth)


def rand_index(a, b):
    if b < a:
        return randint(b, a)
    return randint(a, b)


