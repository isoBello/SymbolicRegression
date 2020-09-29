#!/usr/bin/env python3
import math
from SymbolicRegression import Tree
from random import randint, uniform


class Node:
    def __init__(self, no, value, height, left=None, right=None):
        self.type = no
        self.value = value
        self.height = height
        self.left = left
        if no == Tree.operator and 5 <= value <= 6:
            t = randint(0, 1)
            if t == 0:
                value = uniform(Tree.interval_min, Tree.interval_max)
                self.right = Node(Tree.constant, value, height + 1)
            else:
                self.right = Node(Tree.variable, Tree.rand_index(1, Tree.variables_size), height + 1)
            # Used in right leafes with sen or cos values, 'cause they need a constant after showed
            # (i.e we don't have sen + 2, just sen 2)
        else:
            self.right = right

    def evaluation(self, values):
        # Used to evaluate the "value" of the tree
        if self.type == Tree.constant:
            return self.value
        elif self.type == Tree.operator:
            return self.operations(self.value, self.left.evaluation(values),
                                   self.right.evaluation(values))
        elif self.type == Tree.variable:
            return values[self.value - 1]

    @staticmethod
    def operations(function, a, b):
        if function == 1:
            return a + b
        elif function == 2:
            return a - b
        elif function == 3:
            return a * b
        elif function == 4:
            return a / b if b else 0  # Prevents errors from division: num / 0
        elif function == 5:
            return math.sin(a)
        elif function == 6:
            return math.cos(b)

    def tree_mutation(self, max_depth):
        type_mutation = randint(0, 2)
        if type_mutation == 0:  # Mutate into a variable
            self.type = Tree.variable
            self.value = randint(1, Tree.variables_size)
        elif type_mutation == 1:  # Mutate into a constant
            self.type = Tree.constant
            self.value = uniform(-10, 10)
        elif type_mutation == 2:  # Mutate into a function
            if self.height == max_depth:
                new_tree = Tree.tree(self.height, False)
                self.swap_subtree(new_tree)

    def subtree(self, depth):
        if self.height == depth:  # If we asked a subtree who has the same depth and height
            return self
        else:
            side = randint(0, 1)  # 0 we going left-side; 1 we going right-side
        if side == 0:
            if self.left is not None:
                return self.left.subtree(depth)
            elif self.right is not None:
                return self.right.subtree(depth)
            else:
                return self
        else:
            if self.right is not None:
                return self.right.subtree(depth)
            elif self.left is not None:
                return self.left.subtree(depth)
            else:
                return self

    def swap_subtree(self, b):  # Swap between two nodes
        self.__dict__, b.__dict__ = b.__dict__, self.__dict__
