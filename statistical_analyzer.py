import numpy as np
from time import time

from PyQt6.QtCore import QThread, pyqtSignal

class StatisticalAnalyzer(QThread):
    start_signal = pyqtSignal()
    progress_signal = pyqtSignal(tuple)
    end_signal = pyqtSignal()

    def __init__(self, resampled_matrix):
        super(StatisticalAnalyzer, self).__init__()
        self.reference = np.sort(resampled_matrix[0])[::-1]  # Actual reference values
        self.resampled_matrix = resampled_matrix[1:]  # Get everything but the first row from the original unsorted resampled matrix
        self.indexes = np.argsort(resampled_matrix[0])[::-1]
        self.seq_nums = [int(np.floor(i/2)) for i in self.indexes]  # Because each pair of values is a sequence number

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
        prev_p_val = (None, None, None)

        # For each reference value (sorted in descending order)
        for i in np.arange(len(self.reference)):
            ref, seq_num = self.reference[i], self.seq_nums[i]

            positively_correlated = self.indexes[i] % 2 == 0
                
            # Get null distribution, which is the k'th largest value across every row of the resampled (shortcut if k=1)
            relevant_view = self.resampled_matrix[:, i:]
            null_distribution = np.partition(relevant_view, -k)[:, -k] if k > 1 else np.max(relevant_view, axis=1)

            # The p-value is the proportion of null values that are greater than or equal to the reference value
            p_val = (np.sum(null_distribution >= ref) + 1) / (len(null_distribution) + 1)

            # We're done if the p-value is greater than or equal to the threshold   
            if p_val > alpha:
                break

            if prev_p_val[0] is None:
                # If this is the first p-value, add it to the list
                prev_p_val = (p_val, seq_num, positively_correlated)
            else:
                if p_val >= prev_p_val[0]:
                    # If the p-value is greater than or equal to the previous p-value, add to the list and update the previous p-value
                    p_values.append((p_val, seq_num, positively_correlated))
                    prev_p_val = (p_val, seq_num, positively_correlated)
                else:
                    # If the p-value is less than the previous p-value, add the previous p-value to the list and leave the previous p-value as is
                    p_values.append(prev_p_val)
        return p_values
    
    def FWERControl(self):
        """Finds the number of significant sequences using the FWER control method."""
        self.start_signal.emit()
        self.p_values = self.getPValuesFull(alpha=self.alpha)
        self.k = 1
        self.end_signal.emit()
    
    def fdpControl(self, abbreviated=False):
        """Finds the number of significant sequences using the FDP control method."""
        self.start_signal.emit()
        p_val_func = self.getPValuesFull if not abbreviated else None  # Update this
        progress = lambda l, k: int(100 * ((k / self.gamma) - 1) / l)

        progress_str = f"k\t# p-values\t(k / gamma) - 1\n{'-' * 50}\n"
        
        k = 1
        p_values = p_val_func(k=k, alpha=self.alpha)
        progress_str += f"{k}\t{len(p_values)}\t\t{(k / self.gamma) - 1}\n"
        # print(progress_str)
        self.progress_signal.emit((progress(len(p_values), k), progress_str))
        # print(progress(len(p_values), k))
        
        while len(p_values) >= (k / self.gamma) - 1:
            new_k = int(np.ceil((len(p_values) + 1) * self.gamma))  # Optimization: Find the next k instead of incrementing by 1
            k = k + 1 if new_k == k else new_k
            p_values = p_val_func(k=k, alpha=self.alpha)
            progress_str += f"{k}\t{len(p_values)}\t\t{(k / self.gamma) - 1}\n"
            # print(progress_str)
            self.progress_signal.emit((progress(len(p_values), k), progress_str))

        self.k = k
        self.p_values = p_values
        self.end_signal.emit()
        
    
    def getPValueResults(self):
        return self.p_values, self.k
