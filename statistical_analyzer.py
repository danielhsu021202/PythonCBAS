import numpy as np
from time import time

from PyQt6.QtCore import QThread, pyqtSignal

from files import CBASFile
from sequences import SequencesProcessor

import multiprocessing as mp

class HalfMatrixPValueCalculator:
    """
    This class calculates the null distribution using the half-matrix optimization.
    It is parallelized.
    """
    def getPValues(resampled_matrix, references, seq_nums, indexes, k, alpha):
        # Get the null distribution for each reference value
        p_values = []
        prev_p_val = (None, None, None)


        

class StatisticalAnalyzer(QThread):
    start_signal = pyqtSignal()
    progress_signal = pyqtSignal(tuple)
    end_signal = pyqtSignal()

    def __init__(self, resampled_matrix, k_skip: bool, half_matrix: bool, index_dtype=np.uint32):
        super(StatisticalAnalyzer, self).__init__()
        self.reference = np.sort(resampled_matrix[0])[::-1]  # Actual reference values
        self.resampled_matrix = resampled_matrix[1:]  # Get everything but the first row from the original unsorted resampled matrix
        self.indexes = np.argsort(resampled_matrix[0])[::-1]
        self.seq_nums = [int(np.floor(i/2)) for i in self.indexes]  # Because each pair of values is a sequence number

        self.k_skip = k_skip
        self.half_matrix = half_matrix
        self.index_dtype = index_dtype

        self.alpha = None
        self.gamma = None

        self.p_values = None
        self.k = None

    def setParams(self, alpha, gamma):
        self.alpha = alpha
        self.gamma = gamma


    def isAbbreviated(self):
        """Returns True if the matrix is abbreviated, False otherwise."""
        return type(self.resampled_matrix[1][0]) == tuple
    
    def getPValue(self, params):
        """Returns the p-value for a given reference value and sequence number."""
        i, ref, seq_num, k, alpha = params
        print(f"Start {i}")
        relevant_view = self.resampled_matrix[:, i:]
        null_distribution = np.partition(relevant_view, -k)[:, -k] if k > 1 else np.max(relevant_view, axis=1)

        # The p-value is the proportion of null values that are greater than or equal to the reference value
        p_val = (np.sum(null_distribution >= ref) + 1) / (len(null_distribution) + 1)
        print(f"End {i}")
        return (i, p_val, seq_num), p_val > alpha

    
    def getPValuesFull(self, k=1, alpha=0.05):
        p_values = []
        # prev_p_val = (None, None, None)
        prev_p_val = None

        # For each reference value (sorted in descending order)
        for i in np.arange(len(self.reference)):
            ref = self.reference[i]
            
            # Get null distribution, which is the k'th largest value across every row of the resampled (shortcut if k=1)
            relevant_view = self.resampled_matrix[:, i:]
            null_distribution = np.partition(relevant_view, -k)[:, -k] if k > 1 else np.max(relevant_view, axis=1)

            # The p-value is the proportion of null values that are greater than or equal to the reference value
            p_val = (np.sum(null_distribution >= ref) + 1) / (len(null_distribution) + 1)

            # We're done if the p-value is greater than or equal to the threshold   
            if p_val > alpha:
                break

            # if prev_p_val[0] is None:
            if prev_p_val is None:
                # If this is the first p-value, add it to the list
                prev_p_val = p_val
            else:
                if p_val >= prev_p_val:
                    # If the p-value is greater than or equal to the previous p-value, add to the list and update the previous p-value
                    p_values.append(p_val)
                    prev_p_val = p_val
                else:
                    # If the p-value is less than the previous p-value, add the previous p-value to the list and leave the previous p-value as is
                    p_values.append(prev_p_val)

        # Build the sequence number and positively correlated information into triplets
        num_pvalues = len(p_values)
        if num_pvalues < (k / self.gamma) - 1:
            p_value_triplets = []
            for i, p_val in enumerate(p_values):
                # SOME SORT OF OFF BY ONE THING HERE, BUT THE +1 SEEMED NECESSARY TO MATCH RESULTS IN OLD SCRIPT
                seq_num = self.seq_nums[i+1]
                positively_correlated = self.indexes[i+1] % 2 == 0
                p_value_triplets.append((p_val, seq_num, positively_correlated))
            return num_pvalues, p_value_triplets
        else:
            return num_pvalues, None
    
    def FWERControl(self):
        """Finds the number of significant sequences using the FWER control method."""
        self.start_signal.emit()
        self.p_values = self.getPValuesFull(alpha=self.alpha)
        self.k = 1
        self.end_signal.emit()
    
    def fdpControl(self, abbreviated=False):
        """Finds the number of significant sequences using the FDP control method."""
        self.start_signal.emit()
        p_val_func = self.getPValuesFull if not self.half_matrix else lambda k, alpha: HalfMatrixPValueCalculator.getPValues(self.resampled_matrix, self.reference, self.seq_nums, self.indexes, k, alpha)
        progress = lambda l, k: int(100 * ((k / self.gamma) - 1) / l)

        progress_str = f"k\t# p-values\t(k / gamma) - 1\n{'-' * 50}\n"
        
        k = 1
        num_pvalues, p_values = p_val_func(k=k, alpha=self.alpha)
        progress_str += f"{k}\t{num_pvalues}\t\t{(k / self.gamma) - 1}\n"
        # print(progress_str)
        self.progress_signal.emit((progress(num_pvalues, k), progress_str))
        # print(progress(len(p_values), k))
        
        while num_pvalues >= (k / self.gamma) - 1:
            if self.k_skip:
                new_k = int(np.ceil((num_pvalues + 1) * self.gamma))  # Optimization: Find the next k instead of incrementing by 1
                k = k + 1 if new_k == k else new_k
            else:
                k = k + 1
            num_pvalues, p_values = p_val_func(k=k, alpha=self.alpha)
            progress_str += f"{k}\t{num_pvalues}\t\t{(k / self.gamma) - 1}\n"
            # print(progress_str)
            self.progress_signal.emit((progress(num_pvalues, k), progress_str))

        self.k = k
        self.p_values = p_values
        self.end_signal.emit()
        
    
    def getPValueResults(self):
        return self.p_values
    
    def writeSigSeqFile(self, p_values, seq_num_index, counts_dir, resample_dir):
        p_val_mat = []
        for p_val, seq_num, positively_correlated in p_values:
            seq, cont, length, local_num = SequencesProcessor.getSequence(seq_num, seq_num_index, counts_dir)
            p_val_mat.append([p_val, seq, cont, length, local_num, positively_correlated])
        # p_val_mat = np.array(p_val_mat, dtype=object)
        p_val_file = CBASFile("significant_sequences", p_val_mat, type=CBASFile.file_types['SIGSEQS'])
        p_val_file.saveFile(resample_dir)
