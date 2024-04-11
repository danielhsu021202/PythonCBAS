import numpy as np

class StatisticalAnalyzer:

    def __init__(self, resampled_matrix):
        self.resampled_matrix = resampled_matrix

    def isAbbreviated(self):
        """Returns True if the matrix is abbreviated, False otherwise."""
        return type(self.resampled_matrix[1][0]) == tuple
    
    
    def getPValuesFull(self, n=1, threshold=0.05):

        # TODO: Recursive implementation

        full_mat = self.resampled_matrix.copy()  # Copy the matrix to avoid modifying the original
        reference = np.sort(full_mat[0])[::-1]  # Actual reference values
        seq_nums = np.argsort(full_mat[0])[::-1]
        resampled = full_mat[1:]  # Get everything but the first row from the original unsorted resampled matrix
        

        p_values = []
        prev_p_val = (None, None)
        # For each reference value (sorted in descending order)
        for i in np.arange(len(reference)):
            ref = reference[i]
            seq_num = seq_nums[i]
            # Get null distribution, which is the n'th largest value across every row of the resampled (shortcut if n=1)
            null_distribution = np.partition(resampled, -n)[:, -n] if n > 1 else np.max(resampled, axis=1)
            
            # The p-value is the proportion of null values that are greater than or equal to the reference value
            p_val = (np.sum(null_distribution >= ref) + 1) / (len(null_distribution) + 1)

            # We're done if the p-value is greater than or equal to the threshold   
            if p_val > threshold:
                break

            if prev_p_val is (None, None):
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
                
            # Get rid of the first column of resampled matrix
            resampled = resampled[:, 1:]
        
        return p_values
