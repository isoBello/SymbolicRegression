#!/usr/bin/env python3

import random
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
functions_size = 4
depth_max = 6
depth_min = 2

