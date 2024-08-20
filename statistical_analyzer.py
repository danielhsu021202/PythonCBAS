import numpy as np
from time import time

from PyQt6.QtCore import QThread, pyqtSignal

from files import CBASFile
from sequences import SequencesProcessor

import multiprocessing as mp

class ParallelPValueEngine:
    def __init__(self, reference, resampled_matrix, alpha: float, gamma: float, half_matrix: bool, index_dtype):
        self.reference = reference
        # self.resampled_matrix = resampled_matrix
        self.alpha = alpha
        self.gamma = gamma
        self.half_matrix = half_matrix
        self.index_dtype = index_dtype

        self.sorted_resampled_matrix_indices = np.argsort(-resampled_matrix, axis=1)
        self.sorted_resampled_matrix = np.take_along_axis(resampled_matrix, self.sorted_resampled_matrix_indices, axis=1)

        self.sorted_resampled_matrix_indices = self.sorted_resampled_matrix_indices.flatten()
    
    def getPValues(self, k=1):
        p_values = []
        prev_p_value = None

        sorted_resampled_matrix_copy = self.sorted_resampled_matrix.tolist()
        for i in np.arange(len(self.reference)):
            ref = self.reference[i]
            # Get null distribution, which is the k'th column of the sorted_resampled_matrix_copy
            null_distribution = np.array(sorted_resampled_matrix_copy)[:, k-1]

            # The p-value is the proportion of null values that are greater than or equal to the reference value
            p_val = (np.sum(null_distribution >= ref) + 1) / (len(null_distribution) + 1)

            # We're done if the p-value is greater than or equal to the threshold
            if p_val > self.alpha:
                break

            if prev_p_value is None:
                # If this is the first p-value, add it to the list
                prev_p_value = p_val
            else:
                if p_val >= prev_p_value:
                    # If the p-value is greater than or equal to the previous p-value, add to the list and update the previous p-value
                    p_values.append(p_val)
                    prev_p_value = p_val
                else:
                    # If the p-value is less than the previous p-value, add the previous p-value to the list and leave the previous p-value as is
                    p_values.append(prev_p_value)
            
            # Find the index of all the i's in the index matrix. The x'th entry is the index of the item in the x'th row of the sorted matrix we need to delete.
            indices = np.where(self.sorted_resampled_matrix_indices == i)[0]
            print(f"Length of indices: {len(indices)}")
            print(f"Num rows in sorted_resampled_matrix_copy: {len(sorted_resampled_matrix_copy)}")
            print(f"Num cols in sorted_resampled_matrix_copy: {len(sorted_resampled_matrix_copy[0])}")
            # Delete the i'th column from the sorted matrix
            wrap_length = len(sorted_resampled_matrix_copy[0])
            for i, index in enumerate(indices):
                sorted_resampled_matrix_copy[i].pop(index % wrap_length)

        return p_values, []


        # Sort and get indices
        pool = mp.Pool(mp.cpu_count())

        sorted_indicies = np.empty(resampled_matrix.shape, dtype=index_dtype)
        results = pool.starmap(ParallelPValueEngine.sortRow, [(row, i) for i, row in enumerate(resampled_matrix)])
        for i, row, indices in results:
            sorted_indicies[i] = indices
            resampled_matrix[i] = row
        pool.close()
        pool.join()

        deleted = []  # Each row represents a deleted "column" in the matrix

        for i in np.arange(len(reference)):
            ref = reference[i]
            null_distribution = resampled_matrix[:, k]


            

    def sortRow(row, i):
        sorted_row = np.sort(row)
        indices = np.argsort(row)
        return (i, sorted_row, indices)


    def getPValue(relevant_view, ref, n):
        """
        Parallelized; Get the null distirbution, which is the n'th largest of each row, with the relevant matrix being ith column onwards.
        This one parallelizes only the sorting of the relevant view.
        Seems to be slower than the other method, at least for small matrices.
        """

        pool = mp.Pool(mp.cpu_count())
        null_distribution = pool.starmap(ParallelPValueEngine.nth_largest_geq_ref, [(row, ref, n, i) for i, row in enumerate(relevant_view)])
        pool.close()
        pool.join()

        return (np.count_nonzero(null_distribution) + 1) / (len(null_distribution) + 1)

    def nth_largest_geq_ref(arr, ref, n, i) -> bool:
        """
        Returns the n'th largest element in the array.
        """
        print(i)
        list(arr).sort()
        return arr[-n] >= ref


    


    

    
        

class StatisticalAnalyzer(QThread):
    start_signal = pyqtSignal()
    progress_signal = pyqtSignal(tuple)
    end_signal = pyqtSignal()

    def __init__(self, reference_rates, sorting_indices, resampled_matrix, k_skip: bool, half_matrix: bool, parallelize_sort: bool, index_dtype=np.uint32):
        super(StatisticalAnalyzer, self).__init__()
        self.reference = reference_rates
        self.resampled_matrix = resampled_matrix
        self.indexes = sorting_indices

        # If half matrix, it's not duplicated, so each number is the corresponding sequence number
        self.seq_nums = [i if half_matrix else int(np.floor(i/2)) for i in self.indexes]  # Because each pair of values is a sequence number

        # Settings
        self.k_skip = k_skip
        self.half_matrix = half_matrix
        self.parallelize_sort = parallelize_sort
        self.index_dtype = index_dtype

        self.alpha = None
        self.gamma = None

        # Initialize
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
        prev_p_val = None
        signs = []  # For half matrix, the signs of the rates originally, used to determine correlatedness or g1>g2

        # For each reference value (sorted in descending order)
        for i in np.arange(len(self.reference)):
            ref = self.reference[i]
            signs = []
            
            
            #PROBLEM: With sorting, we're no longer deleting the correct entries... we're always deleting the largest, which is incorrect
            p_val = None
            # relevant_view = self.resampled_matrix[:, (int(np.floor(i/2)) if self.half_matrix else i):]  # If using half the matrix, we only "delete" the first column every other iteration.
            relevant_view = self.resampled_matrix[:, i:]

            null_distribution = []
            if self.half_matrix:
                # Take the absolute values of the rows
                for row in relevant_view:
                    # Get the kth largest value in the row, key is absolute value
                    sort_order = np.argsort(np.abs(row))[::-1]
                    sorted_row = row[sort_order]
                    # Append the sign of the largest absolute value
                    signs.append(sorted_row[0] > 0)
                    # Append the kth largest absolute value
                    null_distribution.append(abs(sorted_row[k-1]))
            else:
                # Get null distribution, which is the k'th largest value across every row of the resampled (shortcut if k=1)
                null_distribution = np.partition(relevant_view, -k)[:, -k] if k > 1 else np.max(relevant_view, axis=1)

            # The p-value is the proportion of null values that are greater than or equal to the reference value
            p_val = (np.sum(null_distribution >= ref) + 1) / (len(null_distribution) + 1)
            
                
                
                

            # We're done if the p-value is greater than or equal to the threshold   
            if p_val > alpha:
                break

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
        return p_values, signs

    
    
    def FWERControl(self):
        """Finds the number of significant sequences using the FWER control method."""
        self.start_signal.emit()
        self.p_values = self.getPValuesFull(alpha=self.alpha)
        self.k = 1
        self.end_signal.emit()
    
    def fdpControl(self, abbreviated=False):
        """Finds the number of significant sequences using the FDP control method."""
        self.start_signal.emit()
        p_val_func = self.getPValuesFull 
        progress = lambda l, k: int(100 * ((k / self.gamma) - 1) / l)

        progress_str = f"k\t# p-values\t(k / gamma) - 1\n{'-' * 50}\n"
        
        parallel_engine = None
        if self.parallelize_sort:
            # Initialize the parallel engine
            parallel_engine = ParallelPValueEngine(self.reference, self.resampled_matrix, self.alpha, self.gamma, self.half_matrix, self.index_dtype)


        k = 1
        p_values = []
        signs = []
        p_values, signs = p_val_func(k=k, alpha=self.alpha)
        num_pvalues = len(p_values)
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

            if self.parallelize_sort:
                p_values, signs = parallel_engine.getPValues(k=k)
            else:
                p_values, signs = p_val_func(k=k, alpha=self.alpha)
            num_pvalues = len(p_values)
            progress_str += f"{k}\t{num_pvalues}\t\t{(k / self.gamma) - 1}\n"
            # print(progress_str)
            self.progress_signal.emit((progress(num_pvalues, k), progress_str))

        self.k = k
        self.p_values = p_values
        self.signs = signs
        self.end_signal.emit()
        
    def buildTriplets(self, p_values):
        # Build the sequence number and positively correlated information into triplets
        p_value_triplets = []
        for i, p_val in enumerate(p_values):
            seq_num = None
            positively_correlated = None
            # SOME SORT OF OFF BY ONE THING HERE, BUT THE +1 SEEMED NECESSARY TO MATCH RESULTS IN OLD SCRIPT
            if self.half_matrix:
                seq_num = self.seq_nums[i+1]
                positively_correlated = self.signs[i+1]
            else:
                seq_num = self.seq_nums[i+1]
                positively_correlated = self.indexes[i+1] % 2 == 0
            p_value_triplets.append((p_val, seq_num, positively_correlated))
        return p_value_triplets

    def getPValueResults(self):
        return self.buildTriplets(self.p_values)
    
    def writeSigSeqFile(self, p_values, seq_num_index, counts_dir, resample_dir):
        p_val_mat = []
        for p_val, seq_num, positively_correlated in p_values:
            seq, cont, length, local_num = SequencesProcessor.getSequence(seq_num, seq_num_index, counts_dir)
            p_val_mat.append([p_val, seq, cont, length, local_num, positively_correlated])
        # p_val_mat = np.array(p_val_mat, dtype=object)
        p_val_file = CBASFile("significant_sequences", p_val_mat, type=CBASFile.file_types['SIGSEQS'])
        p_val_file.saveFile(resample_dir)
