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

from files import FileManager




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


class SequenceManager:
    def __init__(self):
        self.next_number = 0
        self.sequences = {}
        

    def registerSequence(self, sequence: tuple):
        """
        Assign a number to each new sequence and store it in the sequences dictionary.
        """
        seq_num = self.sequences.get(sequence, None)
        if seq_num:
            return seq_num
        else:
            self.sequences[sequence] = self.next_number
            self.next_number += 1
            return self.next_number - 1
        

class SequencesProcessor:
    def __init__(self, FILES, ANIMAL_FILE_FORMAT, LANGUAGE):
        self.FILES = FILES
        self.COLUMNS = ANIMAL_FILE_FORMAT
        self.LANGUAGE = LANGUAGE
        # For analysis, 2D array with length of sequence as rows, contingency as columns, and number of sequences as values
        self.sequence_sets = [[set() for _ in np.arange(7)] for _ in np.arange(self.LANGUAGE['MAX_SEQUENCE_LENGTH'])]

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

    def getSequences(self, mat, length):
        """
        Given a matrix and a length, returns a set of all the sequences of that length.
        Straddles sessions if STRADDLE_SESSIONS is True, otherwise only considers sequences within the same session.
        This gets run num_contingencies * MAX_SEQUENCE_LENGTH times per animal.
        """
        sequences = set()
        if not self.LANGUAGE['STRADDLE_SESSIONS']:
            i = 0
            while i <= len(mat) - length:
                last_idx_of_sequence = i + length - 1
                if mat[i, self.COLUMNS['SESSION_NO_COL']] == mat[last_idx_of_sequence, self.COLUMNS['SESSION_NO_COL']]:
                    # If the first and last session numbers of this sequence are the same, add it to the set of sequences
                    sequences.add(tuple(mat[i:i+length, self.COLUMNS['CHOICE_COL']]))
                    i += 1
                else:
                    # Otherwise, we're straddling sessions, so we need to skip to the second session present in the relevant section
                    # Even if there's a session caught in the middle, skipping to second session will just trigger this case again, so it will handle itself
                    relevant_section = mat[i:last_idx_of_sequence+1, self.COLUMNS['SESSION_NO_COL']]
                    i += np.where(relevant_section != relevant_section[0])[0][0]
        else:
            for i in np.arange(len(mat) - length):
                sequences.add(tuple(mat[i:i+length, self.COLUMNS['CHOICE_COL']]))
        return sequences


    def getAllLengthSequences(self, mats: list[tuple[np.array, int]]):
        """
        Takes a list of tuples, each containing a matrix and a contingency number.
        Get all sequences of length up to MAX_SEQUENCE_LENGTH for each contingency.
        This gets run once per animal.
        """
        for mat, cont in mats:
            # print("PROCESSING CONTINGENCY", cont)
            for length in np.arange(1, self.LANGUAGE['MAX_SEQUENCE_LENGTH'] + 1):
                sequences = self.getSequences(mat, length)
                self.sequence_sets[length-1][cont].update(sequences)
                #print(f"Contingency {cont}, Length {length}, Num Sequences: {len(sequences)}")
        

    

    def generateSequenceFiles(self):
        """
        MAIN FUNCTION FOR THIS MODULE
        """
        all_paths = FileManager.unpickle_obj(os.path.join(self.FILES['METADATA'], 'all_paths.pkl'))
        for animal in all_paths:
            mat = self.getMatrix(animal)
            mat = self.collapseModifiers(mat)
            mats_by_cont = self.splitContingency(mat)
            self.getAllLengthSequences(mats_by_cont)
        sequence_counts = [[len(self.sequence_sets[i][j]) for j in np.arange(7)] for i in np.arange(self.LANGUAGE['MAX_SEQUENCE_LENGTH'])]
        print(np.array(sequence_counts))



    
# Generate 1000000 sequences of length 6 and see if registerSequence or registerSequence2 is faster
# sequences = [tuple(np.random.randint(1, 10, 6)) for _ in np.arange(1000000)]
# start = time.time()
# sequenceManager = SequenceManager()
# for sequence in sequences:
#     sequenceManager.registerSequence(sequence)
# print(f"Time for registerSequence: {time.time() - start}")
# start = time.time()
# sequenceManager = SequenceManager()
# for sequence in sequences:
#     sequenceManager.registerSequence2(sequence)
# print(f"Time for registerSequence2: {time.time() - start}")






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


    

