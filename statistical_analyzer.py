import numpy as np
import multiprocessing as mp
from files import CBASFile
import os, sys
from time import time

class StatisticalAnalyzer:

    def __init__(self, resampled_matrix):
        self.reference = np.sort(resampled_matrix[0])[::-1]  # Actual reference values
        self.resampled_matrix = resampled_matrix[1:]  # Get everything but the first row from the original unsorted resampled matrix
        indexes = np.argsort(resampled_matrix[0])[::-1]
        self.seq_nums = [int(np.floor(i/2)) for i in indexes]  # TODO: 2 is the number of groups, modify this to be more general later




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
    
    def getPValuesFullParallel(self, k=1, alpha=0.05):

        cpu_count = 75
        pool = mp.Pool()
        

        def runChunk(chunk_num):
            chunk_start = chunk_num * cpu_count
            chunk_end = chunk_start + cpu_count
            work = [(i, self.reference[i], self.seq_nums[i], k, alpha) for i in np.arange(chunk_start, chunk_end)]
            print("Go")
            results = pool.map(self.getPValue, work)
            
            results_lst = list(results)
            for _, flag in results_lst:
                if flag:
                    return True, results_lst  # Early termination condition met
            return False, results_lst

        all_results = []
        for chunk_num in np.arange(len(self.reference) // cpu_count):
            print(f"Start chunk {chunk_num}")
            alpha_met, results = runChunk(chunk_num)
            all_results.extend(results)
            if alpha_met:
                break

        pool.close()
        pool.join()
        return all_results

        # p_values = mp.Queue()
        # alpha_met = mp.Event()

        # processes = []
        # for i in np.arange(len(self.reference)):
        #     ref, seq_num = self.reference[i], self.seq_nums[i]
        #     if not alpha_met.is_set():
        #         process = mp.Process(target=self.getPValue, args=(i, ref, seq_num, k, alpha, alpha_met, p_values))
        #         processes.append(process)
        #         process.start()
        #     else:
        #         break

        # for process in processes:
        #     process.join()
        
        # results = []
        # while not p_values.empty():
        #     results.append(p_values.get())

        # return results



        # # pool = mp.Pool(10)

        # # result_iter = pool.imap_unordered(self.getPValue, [(i, ref, seq_num, k, alpha) for i, ref, seq_num in zip(np.arange(len(self.reference)), self.reference, self.seq_nums)])
                    

        # # return list(result_iter)
        
    
    def getPValuesFull(self, k=1, alpha=0.05):
        p_values = []
        prev_p_val = (None, None)

        # For each reference value (sorted in descending order)
        for i in np.arange(len(self.reference)):
            ref, seq_num = self.reference[i], self.seq_nums[i]
            # Get null distribution, which is the k'th largest value across every row of the resampled (shortcut if k=1)
            relevant_view = self.resampled_matrix[:, i:]
            null_distribution = np.partition(relevant_view, -k)[:, -k] if k > 1 else np.max(relevant_view, axis=1)

            # The p-value is the proportion of null values that are greater than or equal to the reference value
            p_val = (np.sum(null_distribution >= ref) + 1) / (len(null_distribution) + 1)

            # p_val = self.getPValue(i, ref, seq_num, k)

            # We're done if the p-value is greater than or equal to the threshold   
            if p_val > alpha:
                break

            if prev_p_val[0] is None:
                # If this is the first p-value, add it to the list
                prev_p_val = (p_val, seq_num)
            else:
                if p_val >= prev_p_val[0]:
                    # If the p-value is greater than or equal to the previous p-value, add to the list and update the previous p-value
                    p_values.append((p_val, seq_num))
                    prev_p_val = (p_val, seq_num)
                else:
                    # If the p-value is less than the previous p-value, add the previous p-value to the list and leave the previous p-value as is
                    p_values.append(prev_p_val)
        return p_values
    
    def fdpControl(self, alpha=0.5, gamma=0.05, abbreviated=False):
        """Finds the number of significant sequences using the FDP control method."""
        p_val_func = self.getPValuesFull if not abbreviated else None  # Update this
            
        k = 1
        p_values = p_val_func(k=k, alpha=alpha)
        print(k, len(p_values), (k / gamma) - 1)
        while len(p_values) >= (k / gamma) - 1:
            # k += 1
            k = int(np.ceil((len(p_values) + 1) * gamma))
            p_values = p_val_func(k=k, alpha=alpha)
            print(k, len(p_values), (k / gamma) - 1)
        return p_values, k



if __name__ == "__main__":
    start_time = time()
    print(f"Loding resampled matrix from file...")
    resampled_matrix_f = CBASFile.loadFile(os.path.join('output_hex', 'resampled_mat_1000_samples_cont_1.cbas'))
    print(f"Resampled matrix loaded. Time taken: {time() - start_time}")
    
    start_time = time()
    print("Calculating p-values...")
    resampled_matrix = resampled_matrix_f.getData()
    stats_analyzer = StatisticalAnalyzer(resampled_matrix)

    p_values = stats_analyzer.getPValuesFullParallel(k=2, alpha=0.05)
    print(len(p_values))
    print(f"P-values calculated. Time taken: {time() - start_time}")
    sys.exit()

