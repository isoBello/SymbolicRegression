#!/usr/bin/env python
# -*- coding: UTF-8 -*-
from SymbolicRegression import Node
from random import randint, uniform
import numpy as np
import logging

# Tree characteristics
operator = "Operator"
constant = "Constant"
variable = "Variable"
functions = {1: 'add', 2: 'sub', 3: 'mul', 4: 'div', 5: 'sen', 6: 'cos'}
functions_size = len(functions)
variables = []
variables_size = 0.0
depth_max = 7
depth_min = 2
interval_max = 0.0
interval_min = 0.0


def print_tree(t, level=0):
    str_tree = '  ' * level

    if level == 0:
        str_tree += "\t\troot: "
    else:
        str_tree.replace("root: ", "")

    if t.type == variable:
        str_tree += "x" + str(t.value)
    elif t.type == operator:
        str_tree += str(get_operator(t.value))
    else:
        str_tree += str(f'{t.value:.4g}')

    str_tree += "\n"

    if t.left is not None:
        str_tree += " Nó esquerdo - Altura " + str(t.height) + ": " + print_tree(t.left, level + 1)
    if t.right is not None:
        str_tree += " Nó direito - Altura " + str(t.height) + ":" + print_tree(t.right, level + 1)

    return str_tree


def tree(depth, method, dmax=depth_max):
    # Method defines if it will be a full or grow tree. If True = Full, otherwise = Grow. Based on ramped-half-and-half
    try:
        if depth == dmax:
            return node(depth)
        else:
            type = randint(0, 1)
            if type == 0 and depth >= depth_min and not method:
                return node(depth)
            else:
                operation = randint(1, functions_size)
                left = tree(depth + 1, method)
                right = tree(depth + 1, method)
                format_tree = Node.Node(operator, operation, depth, left, right)
        return format_tree
    except Exception as e:
        logging.exception(e)


def node(depth):
    # Return a new node, who will be const or var
    no = randint(0, 1)

    if no == 0:
        index = rand_index(1, variables_size)
        variables.append('x' + str(index))
        return Node.Node(variable, index, depth)  # variable
    else:
        return Node.Node(constant, uniform(interval_min, interval_max), depth)  # constant


def get_operator(key):
    for k, v in functions.items():
        if k == key:
            return v


def rand_index(a, b):
    if b < a:
        return randint(b, a)
    return randint(a, b)


def fitness_calculator(a, values):
    # Method who calculates the fitness, using the ABS_ERROR method.
    abs_error = 0.0

    for k, v in values.items():
        fit = a.tree.evaluation(v)  # This is the value we get for our tree
        # The key of this dict is the value we looking for
        expected_fitness = k
        abs_error += np.abs(fit - expected_fitness)
    return abs_error
