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


# Labels for referencing the attributes in the data
SEX = 'sex'
LESION = 'lesion'
IMPLANT = 'implant'
GENOTYPE = 'genotype'
COHORT = 'cohort'
NAME = 'name'
NULL = 'NULL'

INFO_FILE = 'anInfo.txt'

ALL_ATTRIBUTES = {SEX, LESION, IMPLANT, GENOTYPE, COHORT, NAME}

# Catalogue for the possible values of each attribute, in numerical order (0, 1, 2, ...)
# For example, since while making the data, we set 0 to be male and 1 to be female, and it
# appears this way in anInfo.txt, we have under SEX, 'Male' in index 0, and 'Female' in index 1.
CATALOGUE = {
    GENOTYPE: ['WT', 'scn2a heterozygous'],
    SEX: ['Male', 'Female'],
    LESION: ['Unlesioned', 'Hippocampus', 'DMS', 'DREADD'],
    IMPLANT: ['Unimplanted', 'Implanted']
}


### LANGUAGE ###
NUM_CHOICES = 6
STRADDLE_SESSIONS = False  # Session straddling not implemented
NUM_MODIFIERS = 1  # Allow for more later
MODIFIER_RANGES = [6]


### FORMAT OF THE ANDATA FILES ###
SESSION_NO_COL = 0
CHOICE_COL = 1
MODIFIER_COL = 3
CONTINGENCY_COL = 2


# Paths
DATA = 'scn2aDataSwapped/'
OUTPUT = 'output/'
METADATA = 'metadata/'
EXPECTED_OUTPUT = 'expected_output_scn2a/'
COHORTS_FILE = os.path.join(METADATA, 'cohorts.txt')
ANIMALS_FILE = os.path.join(METADATA, 'animals.txt')


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


def setLanguage(num_choices, modifier_ranges):
    """Sets the language for the experiment. The language is defined by the number of choices and the number of modifiers."""
    global NUM_CHOICES
    global NUM_MODIFIERS
    global MODIFIER_RANGES
    NUM_CHOICES = num_choices
    NUM_MODIFIERS = len(modifier_ranges)
    MODIFIER_RANGES = modifier_ranges

### SETUP METADATA ###
def setupFiles():
    """
    Sets up the output and metadata folders and ensures they all exist.
    """
    if not os.path.exists(OUTPUT):
        os.makedirs(OUTPUT)
    if not os.path.exists(METADATA):
        os.makedirs(METADATA)
    
def natural_sort(l):
        convert = lambda text: int(text) if text.isdigit() else text
        alphanum_key = lambda key: [convert(c) for c in re.split('([0-9]+)', key)]
        return sorted(l, key=alphanum_key)

def buildCohortInfo():
    """
    Goes through each cohort and assigns a number to each cohort. Write it to the cohorts file in comma-separated format:
        cohort_name, cohort_number

    Simultaneously builds a dictionary of the cohort names and their corresponding numbers and returns it.
    We build the dictionary simultaneously for consistency, so that the file and the dictionary are in sync.
    """
    # Get the list of all the cohort folder and sort by natural order
    cohort_folders = natural_sort([name for name in os.listdir(DATA) if os.path.isdir(os.path.join(DATA, name))])
    # Write the cohort names and their corresponding numbers to the cohorts file
    with open(COHORTS_FILE, 'w') as f:
        for i, cohort in enumerate(cohort_folders):
            f.write(f"{cohort},{i}\n")
    return {cohort: i for i, cohort in enumerate(cohort_folders)}



def buildAnimalInfo(cohort_dict):
    """
    Goes through the cohorts in order of the cohort_dict and processes the animal information for each cohort.
    Assigns each animal a number and writes it to the animals file in comma-separated format:
        animal number, cohort number, <all info in the info file>
    """
    animal_num = 0
    for cohort_name, cohort_num in cohort_dict.items():
        cohort_folder = os.path.join(DATA, cohort_name)
        info_file = os.path.join(cohort_folder, INFO_FILE)
        animal_files = natural_sort([name for name in os.listdir(cohort_folder) if os.path.isfile(os.path.join(cohort_folder, name)) and name != INFO_FILE])
        animal_info_matrix = getMatrix(info_file)
        assert len(animal_files) == len(animal_info_matrix)
        with open(ANIMALS_FILE, 'a') as f:
            for i, animal_file in enumerate(animal_files):
                animal_info = animal_info_matrix[i]
                f.write(f"{animal_num},{cohort_num},{','.join([str(int(x)) for x in animal_info])}\n")
                animal_num += 1

        
    







def getMatrix(file):
    """Takes a text file and returns a numpy matrix"""
    return np.genfromtxt(file, delimiter=',', dtype=int)

def getContingency(mat, x):
    """Takes a matrix and returns the matrix with CONTINGENCY column equal to x"""
    return mat[mat[:, CONTINGENCY_COL] == x]

def collapseModifiers(mat):
    """
    Takes the matrix returns it with the choice and modifier columns combined.
    choice_col + modifier_col * num_choices
    Keeps other columns the same. Result has one less column.
    ASSUMES the order of the columns is session_no, choice, modifier, contingency.
    """
    mat[:, CHOICE_COL] = mat[:, CHOICE_COL] + mat[:, MODIFIER_COL] * NUM_CHOICES
    return np.delete(mat, MODIFIER_COL, 1)

def splitContingency(mat):
    """
    Takes a matrix and retrns a list of tuples, each tuple containing a matrix with a different contingency.
    """
    # Split the matrix horizontally by the contingency column
    by_contingency = np.split(mat, np.where(np.diff(mat[:, CONTINGENCY_COL]))[0] + 1)
    # Return an array of tuples: each tuple contains the sub-matrix with the contingency column removed, as well as the contingency value.
    return [(np.delete(submat, CONTINGENCY_COL, 1), submat[0, CONTINGENCY_COL]) for submat in by_contingency]

def getSequences(mat, length):
    """
    Takes a matrix and returns a sorted list of the sequences of the given length.
    Checks with STRADDLE_SESSIONS to see if it should straddle sessions.
    """
    # Collects windows of the given length and puts it into the ElementFrequencyCounter.
    counter = ElementFrequencyCounter()
    if STRADDLE_SESSIONS:
        for i in np.arange(len(mat) - length):
            counter.push(tuple(mat[i:i+length, CHOICE_COL]))
    else:
        for i in np.arange(len(mat) - length):
            if mat[i, SESSION_NO_COL] == mat[i+length, SESSION_NO_COL]:
                counter.push(tuple(mat[i:i+length, CHOICE_COL]))
    return counter.get_sorted_elements()

if __name__ == "__main__":
    setupFiles()
    cohort_dict = buildCohortInfo()
    buildAnimalInfo(cohort_dict)
    
    sys.exit()

    mat = getMatrix(os.path.join(DATA, 'scn2aCoh1', 'anData0.txt'))
    mat = collapseModifiers(mat)
    # print(splitContingency(mat)[0])
    cohorts = {}
    for i in range(1000000):
        cohorts["cohort" + str(i)] = i
    print(sys.getsizeof(cohorts) / 1024 / 1024 / 1024)
    

