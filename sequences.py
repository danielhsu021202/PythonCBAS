# Imports
import os
import sys
import shutil
import timeit
import pickle
import pandas as pd
import numpy as np
from numpy.lib.stride_tricks import as_strided
import re
import csv
from colorama import Fore
import time


from collections import defaultdict
from random import randint




class ElementFrequencyCounter:
    """Default dict that counts the frequency of each element pushed into it. Can retrieve the elements in sorted order."""
    def __init__(self):
        self.counter = defaultdict(int)

    def push(self, item):
        """Push the item to the dict and increment its count. If the item is not in the dict, it will be added."""
        self.counter[item] += 1

    def get_sorted_elements(self):
        sorted_elements = sorted(self.counter, key=lambda x: self.counter[x], reverse=True)
        return sorted_elements
    
    def merge(self, other_counter):
        self.counter.update(other_counter.counter)

    def merge_counters(counters):
        """
        Merges multiple counters into a single counter.
        """
        merged_counter = ElementFrequencyCounter()

        for counter in counters:
            merged_counter.merge(counter)

        return merged_counter


class SequencesProcessor:
    def __init__(self, ANIMAL_FILE_FORMAT, LANGUAGE):
        self.COLUMNS = ANIMAL_FILE_FORMAT
        self.LANGUAGE = LANGUAGE

    def getMatrix(self, file):
        """Takes a text file and returns a numpy matrix"""
        return np.genfromtxt(file, delimiter=',', dtype=int)

    def getContingency(self, mat, x):
        """Takes a matrix and returns the matrix with CONTINGENCY column equal to x"""
        return mat[mat[:, self.COLUMNS['CONTINGENCY_COL']] == x]

    def collapseModifiers(self, mat):
        """
        Takes the matrix returns it with the choice and modifier columns combined.
        choice_col + modifier_col * num_choices
        Keeps other columns the same. Result has one less column.
        ASSUMES the order of the columns is session_no, choice, modifier, contingency.
        """
        mat[:, self.COLUMNS['CHOICE_COL']] = mat[:, self.COLUMNS['CHOICE_COL']] + mat[:, self.COLUMNS['MODIFIER_COL']] * self.LANGUAGE['NUM_CHOICES']
        return np.delete(mat, self.COLUMNS['MODIFIER_COL'], 1)

    def splitContingency(self, mat):
        """
        Takes a matrix and retrns a list of tuples, each tuple containing a matrix with a different contingency.
        """
        # Split the matrix horizontally by the contingency column
        by_contingency = np.split(mat, np.where(np.diff(mat[:, self.COLUMNS['CONTINGENCY_COL']]))[0] + 1)
        # Return an array of tuples: each tuple contains the sub-matrix with the contingency column removed, as well as the contingency value.
        return [(np.delete(submat, self.COLUMNS['CONTINGENCY_COL'], 1), submat[0, self.COLUMNS['CONTINGENCY_COL']]) for submat in by_contingency]

# def getSequences(mat, length):
#     """
#     Takes a matrix and returns a sorted list of the sequences of the given length.
#     Checks with STRADDLE_SESSIONS to see if it should straddle sessions.
#     """
#     # Collects windows of the given length and puts it into the ElementFrequencyCounter.
#     counter = ElementFrequencyCounter()
#     if LANGUAGE['STRADDLE_SESSIONS']:
#         for i in np.arange(len(mat) - length):
#             counter.push(tuple(mat[i:i+length, ANIMAL_FILE_FORMAT['CHOICE_COL']]))
#     else:
#         for i in np.arange(len(mat) - length):
#             if mat[i, ANIMAL_FILE_FORMAT['SESSION_NO_COL']] == mat[i+length, ANIMAL_FILE_FORMAT['SESSION_NO_COL']]:
#                 counter.push(tuple(mat[i:i+length, ANIMAL_FILE_FORMAT['CHOICE_COL']]))
#     return counter.get_sorted_elements()


    

