import numpy as np
import multiprocessing

class StatisticalAnalyzer:

    def __init__(self, resampled_matrix):
        self.reference = np.sort(resampled_matrix[0])[::-1]  # Actual reference values
        self.resampled_matrix = resampled_matrix[1:]  # Get everything but the first row from the original unsorted resampled matrix
        indexes = np.argsort(resampled_matrix[0])[::-1]
        self.seq_nums = [int(np.floor(i/2)) for i in indexes]  # TODO: 2 is the number of groups, modify this to be more general later
        resampled_matrix = None  # Clear the original matrix to save memory


    def isAbbreviated(self):
        """Returns True if the matrix is abbreviated, False otherwise."""
        return type(self.resampled_matrix[1][0]) == tuple
    
    def getPValue(self, i, ref, seq_num, k, alpha):
        """
        Returns the p-value for a given reference value.
        Used for parallel processing, so will return the p-value and the sequence number, and whether it makes the threshold. 
        """
        relevant_view = self.resampled_matrix[:, i:]
        null_distribution = np.partition(relevant_view, -k)[:, -k] if k > 1 else np.max(relevant_view, axis=1)
        p_val = (np.sum(null_distribution >= ref) + 1) / (len(null_distribution) + 1)

        return p_val, seq_num, p_val <= alpha
    
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

        
    # def getPValuesFull(self, k=1, alpha=0.05):

    #     # TODO: Recursive implementation

    #     full_mat = self.resampled_matrix.copy()  # Copy the matrix to avoid modifying the original
    #     reference = np.sort(full_mat[0])[::-1]  # Actual reference values
    #     indexes = np.argsort(full_mat[0])[::-1]
    #     seq_nums = [int(np.floor(i/2)) for i in indexes]  # TODO: 2 is the number of groups, modify this to be more general later
    #     resampled = full_mat[1:]  # Get everything but the first row from the original unsorted resampled matrix
        

    #     p_values = []
    #     prev_p_val = (None, None)
    #     # For each reference value (sorted in descending order)
    #     for i in np.arange(len(reference)):
    #         ref = reference[i]
    #         seq_num = seq_nums[i]
    #         # Get null distribution, which is the n'th largest value across every row of the resampled (shortcut if n=1)
    #         null_distribution = np.partition(resampled, -k)[:, -k] if k > 1 else np.max(resampled, axis=1)
            
    #         # The p-value is the proportion of null values that are greater than or equal to the reference value
    #         p_val = (np.sum(null_distribution >= ref) + 1) / (len(null_distribution) + 1)

    #         # We're done if the p-value is greater than or equal to the threshold   
    #         if p_val > alpha:
    #             break

    #         if prev_p_val == (None, None):
    #             # If this is the first p-value, add it to the list
    #             prev_p_val = (p_val, seq_num)
    #         else:
    #             if p_val >= prev_p_val[0]:
    #                 # If the p-value is greater than or equal to the previous p-value, add to the list and update the previous p-value
    #                 p_values.append((p_val, seq_num))
    #                 prev_p_val = (p_val, seq_num)
    #             else:
    #                 # If the p-value is less than the previous p-value, add the previous p-value to the list and leave the previous p-value as is
    #                 p_values.append(prev_p_val)
                
    #         # Get rid of the nth column of resampled matrix
    #         resampled = np.delete(resampled, -k, axis=1)
        
    #     return p_values
    
    def fdpControl(self, alpha=0.5, gamma=0.05, abbreviated=False):
        """Finds the number of significant sequences using the FDP control method."""
        p_val_func = self.getPValuesFull if not abbreviated else None  # Update this
            
        k = 1
        p_values = p_val_func(k=k, alpha=alpha)
        print(len(p_values), (k / gamma) - 1)
        while len(p_values) >= (k / gamma) - 1:
            k += 1
            p_values = p_val_func(k=k, alpha=alpha)
            print(len(p_values), (k / gamma) - 1)
        return p_values, k
