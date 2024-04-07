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


from random import randint

from files import FileManager, CBASFile
from settings import Settings
from utils import HexUtils, FileUtils, MatrixUtils



class SequenceManager:
    """
    Manages the sequence numbers as well as the trial numbers and animal numbers of each sequence.
    Each SequenceManager object corresponds to a single sequence length and contingency.
    """
    universal_next_number = 0  # The next number to assign to a new sequence

    def __init__(self, cont: int, length: int):

        self.seq_nums = {}  # Maps sequences to numbers
        self.animal_trials = []  # This is a 2D array for allSeqAllAn
        self.seq_counts = np.zeros((1, 1)) # This is a 2D array for seqCnts


        self.cont = cont
        self.length = length
        
    def getSeqNums(self):
        return self.seq_nums
    
    def getAnimalTrials(self):
        return self.animal_trials
    
    def getSeqCounts(self):
        return self.seq_counts

    def registerSeqNum(self, sequence: tuple):
        """
        Assign a number to each new sequence and store it in the sequences dictionary.
        """
        seq_num = self.seq_nums.get(sequence, None)
        if seq_num is not None:
            return seq_num
        else:
            # seq_num = HexUtils.getSeqNumHexNum(self.next_number, self.cont, self.length, self.LANGUAGE)
            self.seq_nums[sequence] = SequenceManager.universal_next_number
            SequenceManager.universal_next_number += 1
            return SequenceManager.universal_next_number - 1
        
    def registerAnimalAndTrial(self, animal_num: int, trial_num: int, seq_num: int):
        """
        Register the animal number and trial number of a sequence.
        """
        self.animal_trials.append([animal_num, trial_num, seq_num])
    
    def registerSequence(self, sequence: tuple, animal_num: int, trial_num: int):
        """
        Register a sequence and return its number.
        """
        seq_num = self.registerSeqNum(sequence)
        self.registerAnimalAndTrial(animal_num, trial_num, seq_num)
        return seq_num
        
    def genAllSeqFile(self, FILES, cont, seq_len):
        """
        Generate the allSeq file for this sequence length and contingency.
        """
        # Turn into a 2D array
        mat = np.array([[num for num in sequence] for sequence in self.seq_nums.keys()])
        FileUtils.writeMatrix(os.path.join(FILES['ALLSEQDIR'], f'allSeq_{cont}_{seq_len}.txt'), mat)
    
    def genAllSeqAllAnFile(self, FILES, cont, seq_len):
        """
        Generate the allSeqAllAn file for this sequence length and contingency.
        """
        # Turn into a 2D array
        mat = np.array(self.animal_trials)
        FileUtils.writeMatrix(os.path.join(FILES['ALLSEQALLANDIR'], f'allSeqAllAn_{cont}_{seq_len}.txt'), mat)

    
    def numUniqueSeqsUniversal(self):
        """Returns the largest sequence number registered."""
        return SequenceManager.universal_next_number - 1

    def __len__(self):
        """Returns the number of sequences registered, not unique."""
        return len(self.animal_trials)
    
    
class SequenceCountsProcessor: 
    def __init__(self, animal_count: int, CONSTANTS):
        self.animal_count = animal_count
        self.seq_counts = np.zeros((self.animal_count, 1))
        self.CONSTANTS = CONSTANTS
        
    def updateSeqCnt(self, seq_num: int, animal_num: int, increment: bool):
        """
        Increment the sequence counts for this sequence length and contingency.
        The sequence counts matrix is animal numbers as rows by sequence numbers as columns.
        """
        # If seq_num is greater than or equal to the number of columns, we double the columns or add enough columns to accommodate the new sequence number, whichever is greater
        rows, cols = self.seq_counts.shape
        if seq_num >= cols:
            cols_to_add = max(seq_num - cols + 1, cols * 2)
            self.seq_counts = np.append(self.seq_counts, np.zeros((self.animal_count, cols_to_add)), axis=1)
        # If animal_num is greater than or equal to the number of rows, we add enough rows to accommodate the new animal number
        # if animal_num >= rows:
        #     self.seq_counts = np.append(self.seq_counts, np.zeros((animal_num - rows + 1, len(self.seq_counts[0]))), axis=0)
        # Increment the count
        if increment:
            self.seq_counts[animal_num][seq_num] += 1

    def nanAnimalSeqCnts(self, seq_num: int, animal_nums: list[int]):
        """
        Set the columns in the seq_num and rows in animal_nums to NaN.
        """
        for animal_num in animal_nums:
            self.seq_counts[animal_num][seq_num] = self.CONSTANTS['NaN']

    def setParticipationSeqCnts(self, seq_range: tuple, animal_nums: list[int], null_val: int):
        """
        Set the columns in the seq_range and rows in animal_nums to null_val.
        """
        for animal_num in animal_nums:
            self.seq_counts[animal_num][seq_range[0]:seq_range[1]] = null_val
        # null_array = np.full(len(self.seq_counts[0]), null_val)
        # for animal_num in animal_nums:
        #     self.seq_counts[animal_num] = null_array
    
    def trimSeqCnts(self, next_number: int):
        """
        Trim columns with index greater than the maximum sequence number.
        """
        self.seq_counts = self.seq_counts[:, :next_number]
        # self.seq_counts = self.seq_counts[:, np.any(self.seq_counts, axis=0)]

    def genSeqCntsFile(self, FILES):
        """
        Generate the seqCnts file for this sequence length and contingency.
        """
        # name = f'seqCnts'
        # cbasFile = CBASFile(name, self.seq_counts)
        # cbasFile.saveFile(FILES['SEQCNTSDIR'])
        FileUtils.writeMatrix(os.path.join(FILES['SEQCNTSDIR'], f'seqCnts.txt'), self.seq_counts)
        

class SequencesProcessor:
    def __init__(self, settings:Settings):
        # Import Settings
        self.FILES = settings.getFiles()
        self.COLUMNS = settings.getAnimalFileFormat()
        self.LANGUAGE = settings.getLanguage()
        self.CRITERION = settings.getCriterion()
        self.CONSTANTS = settings.getConstants()

        # For analysis, 2D array with length of sequence as rows, contingency as columns, and number of sequences as values
        self.sequence_matrix = [[SequenceManager(cont, seq_len+1) for cont in np.arange(self.LANGUAGE['NUM_CONTINGENCIES'])] for seq_len in np.arange(self.LANGUAGE['MAX_SEQUENCE_LENGTH'])]
        self.seq_cnts_processor = None  # Will initialize once we get the total number of animals
        self.criterion_matrix = None  # This is initialized in the processAllAnimals() function
        self.seq_rates_df = None  # This is initialized in the getSequenceRates() function
        self.seq_num_index = {}  # Maps sequence number range to (cont, length) pair
        self.current_animal_num = self.CONSTANTS['NaN']
        self.total_animals = self.CONSTANTS['NaN']

        self.missing_contingencies = {}  # Maps contingency to a list of animals that did not participate in that contingency

    def getSequenceManager(self, cont: int, length: int) -> SequenceManager:
        """
        Given a contingency and a length, return the SequenceManager object corresponding to that contingency and length.
        """
        return self.sequence_matrix[length-1][cont]
    
    def getSequenceMatrix(self):
        return self.sequence_matrix

    def registerToSeqMatrix(self, sequence: tuple, length: int, cont: int, trial_num: int) -> int:
        """
        Given a sequence, its length, and its contingency, register it in the sequence_matrix.
        Returns the sequence number of the sequence.
        """
        return self.sequence_matrix[length-1][cont].registerSequence(sequence, self.current_animal_num, trial_num)
    
    def updateCriterionMatrix(self, cont: int, new_trial_num: int, sequence: tuple, orderZero = False):
        """
        Given an animal number, a contingency, and the number of trials to reach the criterion, register it in the criterion_matrix.
        Returns whether the criterion has been reached (which means we've got our trial number).
            If True, then we can stop updating the sequence counts.
            If False, then we need to keep updating the sequence counts.

        criterion matrix: 2D array with animal numbers as rows and contingencies as columns
            Each entry is a tuple: (number accomplished, trial number of accomplishment)
            SPECIAL CASE FOR ORDER 0: If the order is 0, then the first number is the total number of trials performed.
        """
        if new_trial_num is None:
            # If the trial number is None, then the animal did not participate in this contingency
            self.criterion_matrix[self.current_animal_num][cont] = (self.CONSTANTS['NaN'], self.CONSTANTS['NaN'])
            return False
        if orderZero:
            # If we're working with order 0, then we're just registering the total number of trials performed
            self.criterion_matrix[self.current_animal_num][cont] = (min(new_trial_num, self.CRITERION['NUMBER']), self.CONSTANTS['NaN'])
            return new_trial_num >= self.CRITERION['NUMBER']
        else:
            num_accomplished, _ = self.criterion_matrix[self.current_animal_num][cont]    
            if num_accomplished == self.CRITERION['NUMBER']: 
                return True # If the criterion number has already been reached, we're done
            if self.perfectPerformance(sequence):
                # If the sequence performed is perfect, increment the criterion matrix.
                self.criterion_matrix[self.current_animal_num][cont] = (num_accomplished + 1, new_trial_num)
                return False
            
    def perfectPerformance(self, sequence: tuple):
        """
        Return whether the sequence is exclusively rewarded, meaning each number is greater than or equal to NUM_CHOICES
        """
        if not sequence:
            return False
        return all([num >= self.LANGUAGE['NUM_CHOICES'] for num in sequence])

    def postProcessCriterionMatrixCurrAnimalRow(self):
        """
        Post-process the criterion matrix to exclude animals that did not reach the criterion.
        Turns the tuples into a single number, inf if the criterion was not reached, and 0 if the subject did not participate in that contingency.
        Does this for a single row of the criterion matrix corresponding to the current animal being processed.
        """
        if self.CRITERION['ORDER'] == 0:
            for cont in np.arange(self.LANGUAGE['NUM_CONTINGENCIES']):
                self.criterion_matrix[self.current_animal_num][cont] = self.criterion_matrix[self.current_animal_num][cont][0]
        else:
            for cont in np.arange(self.LANGUAGE['NUM_CONTINGENCIES']):
                num_accomplished, trial_num = self.criterion_matrix[self.current_animal_num][cont]
                if num_accomplished == self.CONSTANTS['NaN']:
                    self.criterion_matrix[self.current_animal_num][cont] = self.CONSTANTS['NaN']
                elif num_accomplished < self.CRITERION['NUMBER']:
                    self.criterion_matrix[self.current_animal_num][cont] = self.CONSTANTS['inf']
                else:
                    self.criterion_matrix[self.current_animal_num][cont] = trial_num

    def findCriterionTrial(self, cont: int):
        """
        Effectively post processes the criterion matrix to find the trial number of the criterion for the current animal and contingency.
        """
        num_accomplished, trial_num = self.criterion_matrix[self.current_animal_num][cont]
        if self.CRITERION['ORDER'] == 0:
            if num_accomplished == self.CONSTANTS['NaN']:
                return self.CONSTANTS['NaN']
            return self.CRITERION['NUMBER']
        else:
            if num_accomplished == self.CONSTANTS['NaN']:
                return self.CONSTANTS['NaN']
            elif num_accomplished < self.CRITERION['NUMBER']:
                return float('inf')
            else:
                return trial_num


    def updateSequenceCounts(self, seq_num: int, increment: bool):
        """
        Increment the sequence counts for this sequence length and contingency.
        If increment is False, we are just registering the sequence number and animal number to the sequence counts matrix, so no incrementing needed.
        """
        self.seq_cnts_processor.updateSeqCnt(seq_num, self.current_animal_num, increment)

    def registerMissingContingency(self, cont: int):
        """
        Register the missing contingency for the current animal.
        """
        if cont in self.missing_contingencies:
            self.missing_contingencies[cont].append(self.current_animal_num)
        else:
            self.missing_contingencies[cont] = [self.current_animal_num]

    def getMatrix(self, file):
        """Takes a text file and returns a numpy matrix, enforcing 2D where if there's only one column, each row is its own subarray"""
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
        WORKS BEST IF THE MATRIX IS SORTED BY CONTINGENCY
        """
        # Process missing contingencies
        contingencies = set(np.unique(mat[:, self.COLUMNS['CONTINGENCY_COL']]))
        all_contingencies = set(np.arange(self.LANGUAGE['NUM_CONTINGENCIES']))
        # Find the missing contingencies
        missing_contingencies = all_contingencies - contingencies
        for missing_contingency in missing_contingencies:
            # Set the trial # to -1 for in the criterion_matrix
            self.updateCriterionMatrix(missing_contingency, None, None)
            self.registerMissingContingency(missing_contingency)  # Register the missing contingency for the current animal
        # Split the matrix horizontally by the contingency column
        by_contingency = np.split(mat, np.where(np.diff(mat[:, self.COLUMNS['CONTINGENCY_COL']]))[0] + 1)
        # Return an array of tuples: each tuple contains the sub-matrix with the contingency column removed, as well as the contingency value.
        return [(np.delete(submat, self.COLUMNS['CONTINGENCY_COL'], 1), submat[0, self.COLUMNS['CONTINGENCY_COL']]) for submat in by_contingency]

    def getAllLengthSequences(self, mats: list[tuple[np.array, int]]):
        """
        Takes a list of tuples: (mat, cont)
        Get all sequences of length up to MAX_SEQUENCE_LENGTH for each contingency.
        This gets run once per animal.
        """
        for mat, cont in mats:
            lengths = set(np.arange(1, self.LANGUAGE['MAX_SEQUENCE_LENGTH'] + 1))
            if self.CRITERION['ORDER'] == 0:
                # Then pass in the total number of trials performed
                self.updateCriterionMatrix(cont, len(mat), None, orderZero=True)
            else:
                # We process the sequences corresponding to the criterion order first
                self.getSequences(mat, self.CRITERION['ORDER'], cont, is_order_length=True)
                lengths.remove(self.CRITERION['ORDER'])
            # For each remaining length, get all the sequences
            for length in lengths:
                self.getSequences(mat, length, cont, is_order_length=False)

    def getSequences(self, mat, length: int, cont: int, is_order_length: bool):
        """
        Given a matrix and a length, returns a set of all the sequences of that length.
        Straddles sessions if STRADDLE_SESSIONS is True, otherwise only considers sequences within the same session.
        Processes criterion if is_order_length is True.
        Increments the sequence counts matrix while we are on trial up to and including the criterion trial.
        This gets run num_contingencies * MAX_SEQUENCE_LENGTH times per animal.
        """
        if not self.LANGUAGE['STRADDLE_SESSIONS']:
            trial = 0  # We uniquely identify sequences by their the first trial of occurence

            # If for this contingency, there are animals that did not participate, we'll go ahead and set their sequence counts to NaN
            animals_missing_this_cont = self.missing_contingencies[cont] if cont in self.missing_contingencies else []

            while trial <= len(mat) - length:
                last_idx_of_sequence = trial + length - 1
                if mat[trial, self.COLUMNS['SESSION_NO_COL']] == mat[last_idx_of_sequence, self.COLUMNS['SESSION_NO_COL']]:
                    # If the first and last session numbers of this sequence are the same, add it to the set of sequences
                    sequence = tuple(mat[trial:trial+length, self.COLUMNS['CHOICE_COL']])
                    seq_num = self.registerToSeqMatrix(sequence, length, cont, trial)
                    
                    # Updating the criterion matrix if we're on the criterion order
                    if is_order_length:
                        criterion_reached = self.updateCriterionMatrix(cont, trial, sequence)
                        # If the criterion has not been reached, we need to keep updating the sequence counts
                        self.updateSequenceCounts(seq_num, not criterion_reached)
                    else:
                        # Otherwise, the criterion has already been processed and we can now use the criterion trial number to update the sequence counts
                        self.updateSequenceCounts(seq_num, trial <= self.findCriterionTrial(cont))

                    if animals_missing_this_cont:
                        self.seq_cnts_processor.nanAnimalSeqCnts(seq_num, animals_missing_this_cont)

                    trial += 1
                else:
                    # Otherwise, we're straddling sessions, so we need to skip to the second session present in the relevant section
                    # Even if there's a session caught in the middle, skipping to second session will just trigger this case again, so it will handle itself
                    relevant_section = mat[trial:last_idx_of_sequence+1, self.COLUMNS['SESSION_NO_COL']]
                    trial += np.where(relevant_section != relevant_section[0])[0][0]
        else:
            for i in np.arange(len(mat) - length):
                sequence = tuple(mat[i:i+length, self.COLUMNS['CHOICE_COL']])
                self.registerToSeqMatrix(sequence, length, cont)
    
    

    def processAllAnimals(self):
        """
        ONE OF MAIN CALLED FUNCTION FOR THIS MODULE
        Goes through all the animals and processes their sequences, registering them to the SequenceManager contained in 
        self.sequence_matrix.
        """
        all_paths = FileUtils.unpickleObj(os.path.join(self.FILES['METADATA'], 'all_paths.pkl'))
        self.total_animals = len(all_paths)
        self.seq_cnts_processor = SequenceCountsProcessor(self.total_animals, self.CONSTANTS)
        self.criterion_matrix = [[(0, 0) for _ in np.arange(self.LANGUAGE['NUM_CONTINGENCIES'])] for _ in np.arange(self.total_animals)]
        for animal_num, animal in enumerate(all_paths):
            self.current_animal_num = animal_num
            mat = self.getMatrix(animal)
            mat = self.collapseModifiers(mat)
            mats_by_cont = self.splitContingency(mat)
            self.getAllLengthSequences(mats_by_cont)
            self.postProcessCriterionMatrixCurrAnimalRow()
        
        # POSTPROCESS
        self.seq_cnts_processor.trimSeqCnts(SequenceManager.universal_next_number)

        # for length in np.arange(self.LANGUAGE['MAX_SEQUENCE_LENGTH']):
        #     for cont in np.arange(self.LANGUAGE['NUM_CONTINGENCIES']):
        #         # Build the sequence number index
        #         seq_nums = self.sequence_matrix[length][cont].getSeqNums().values()
        #         self.seq_num_index[(min(seq_nums), max(seq_nums))] = (cont, length+1)

        # # Set the missing contingencies to NaN in the sequence counts matrices
        # for missing_cont, animal_lst in self.missing_contingencies.items():
        #     # Get the ranges involved with this contingency using the sequence number index
        #     for seq_range, cont_len_pair in self.seq_num_index.items():
        #         if cont_len_pair[0] == missing_cont:
        #             self.seq_cnts_processor.setParticipationSeqCnts(seq_range, animal_lst, self.CONSTANTS['NaN'])


            # # Get all the SequenceManager objects for this contingency
            # col = [self.sequence_matrix[length][missing_cont] for length in np.arange(self.LANGUAGE['MAX_SEQUENCE_LENGTH'])]
            # for seq_manager in col:
            #     seq_manager.setParticipationSeqCnts(self.missing_contingencies[missing_cont], self.CONSTANTS['NaN'])



        return self.sequence_matrix

    def generateSequenceFiles(self):
        """
        ONE OF MAIN CALLED FUNCTION FOR THIS MODULE
        Generates the allSeq and allSeqAllAn files for each sequence length and contingency.
        Also generates the criterionMatrix file.
        """
        for length in np.arange(self.LANGUAGE['MAX_SEQUENCE_LENGTH']):
            for cont in np.arange(self.LANGUAGE['NUM_CONTINGENCIES']):
                self.sequence_matrix[length][cont].genAllSeqFile(self.FILES, cont, length+1)
                self.sequence_matrix[length][cont].genAllSeqAllAnFile(self.FILES, cont, length+1)
        
        # Write the sequence counts file
        self.seq_cnts_processor.genSeqCntsFile(self.FILES)
        
        # Write the criterion matrix to a file
        FileUtils.writeMatrix(os.path.join(self.FILES['OUTPUT'], 
                                             f'criterionMatrix_{self.CRITERION["ORDER"]}_{self.CRITERION["NUMBER"]}_{self.CRITERION["INCLUDE_FAILED"]}_{self.CRITERION["ALLOW_REDEMPTION"]}.txt'), 
                                self.criterion_matrix)
        
        # Pickle the sequence number index into metadata
        FileUtils.pickleObj(self.seq_num_index, os.path.join(self.FILES['METADATA'], 'seq_num_index.pkl'))
        
        
    def exportAllSeqCnts(self):
        """
        Creates a lengths x contingencies matrix that contains the sequence counts matrix for each length and contingency.
        """
        all_seq_cnts = np.zeros((self.LANGUAGE['MAX_SEQUENCE_LENGTH'], self.LANGUAGE['NUM_CONTINGENCIES']), dtype=object)
        for length in np.arange(self.LANGUAGE['MAX_SEQUENCE_LENGTH']):
            for cont in np.arange(self.LANGUAGE['NUM_CONTINGENCIES']):
                all_seq_cnts[length][cont] = self.sequence_matrix[length][cont].getSeqCounts()
        return all_seq_cnts


    
        


    

