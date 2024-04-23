import os
import numpy as np
from random import choices, seed
import pandas as pd
from settings import Settings
from utils import FileUtils
from files import CBASFile
import numpy as np
import multiprocessing



class Resampler:
    def __init__(self, settings: Settings, conts='all', seed=925):
        self.LANGUAGE = settings.getLanguage()
        self.CONSTANTS = settings.getConstants()
        self.ANINFOFORMAT = settings.getAnimalInfoFormat()
        self.FILES = settings.getFiles()
        self.CRITERION = settings.getCriterion()
        self.conts = conts
        self.seed = seed
        self.all_seqcnts_matrix = None
        self.all_animals = None
        self.orig_groups = None

    def setAllSeqCntsMatrix(self, all_seqcnts_matrix):
        if self.conts == 'all':
            self.all_seqcnts_matrix = all_seqcnts_matrix
        else:
            self.all_seqcnts_matrix = all_seqcnts_matrix[:, self.conts]
        

    def assignGroups(self, filters: list[dict]):
        """Assigns animals to groups based on the filters provided."""
        animal_matrix = pd.read_csv(self.FILES['ANIMALS_FILE'], header=None)
        an_nums = []
        for filter in filters:
            animal_matrix_copy = animal_matrix.copy()
            # Each element in the group is a filter
            for col_name, value in filter.items():
                col_num = self.ANINFOFORMAT[col_name] + 2  # +2 because the first two columns are animal number and cohort number
                animal_matrix_copy = animal_matrix_copy[animal_matrix_copy[col_num] == value]
            an_nums.append(list(animal_matrix_copy[0]))
        self.orig_groups = an_nums
        self.all_animals = [animal for group in self.orig_groups for animal in group]
    
    def resampleGroups(self, id):
        """Resamples the groups to create new groups of animals."""
        seed(int(id))
        new_groups = []
        for group in self.orig_groups:
            new_group = choices(self.all_animals, k=len(group))
            new_groups.append(new_group)
        return new_groups
    
    def totalSeqs(self):
        """Returns the total number of sequences in the all_seqcnts_matrix."""
        num_seqs = 0
        lengths, conts = self.all_seqcnts_matrix.shape
        for length in np.arange(lengths):
            for cont in np.arange(conts):
                num_seqs += pd.DataFrame(self.all_seqcnts_matrix[length][cont]).shape[1]
        return num_seqs

        
    def getSequenceRatesVerbose(self, groups: list[np.array]):
        """
        Given a list of groups of sequences, returns the sequence rates for each group.
        The columns are:
            Group 1 Averages ... Group n Averages, Original Sequence Number, Length, Contingency
        The rows are sequences, regardless of length and contingency.
        """
        seq_num_col_name = "Original Seq No."
        seq_rates_df = pd.DataFrame()
        lengths, conts = self.all_seqcnts_matrix.shape
        for cont in np.arange(conts):
            for length in np.arange(lengths):
                seq_cnts = pd.DataFrame(self.all_seqcnts_matrix[length][cont])
                one_cont_one_len_df = pd.DataFrame()
                for i, group in enumerate(groups):
                    seq_cnts_g = seq_cnts.iloc[group]  # Keep only the animals (rows) that are in the group
                    seq_cnts_g = seq_cnts_g.loc[~(seq_cnts_g == self.CONSTANTS['NaN']).any(axis=1)]  # Remove animals (rows) with -1 (NaN) in the sequence counts
                    seq_cnts_g = seq_cnts_g.T  # Transpose so that sequences are rows and animals are columns
                    variance = seq_cnts_g.var(axis=1)  # Calculate the variance of the sequence counts (each row) for each sequence
                    n = seq_cnts_g.shape[1]  # Number of animals in the group
                    seq_cnts_g = seq_cnts_g.mean(axis=1)  # Average the sequence counts (each row) for each sequence
                    seq_cnts_g = seq_cnts_g.reset_index()
                    seq_cnts_g.columns = [seq_num_col_name, f'Group {i+1} Avg']
                    seq_cnts_g[f'Group {i+1} Var'] = variance
                    seq_cnts_g[f'Group {i+1} N'] = n # Number of animals in the group

                    # Combine the dataframes from each group
                    if one_cont_one_len_df.empty:
                        one_cont_one_len_df = seq_cnts_g
                    else:
                        one_cont_one_len_df = pd.merge(one_cont_one_len_df, seq_cnts_g, on=seq_num_col_name)
                
                # Add the contingency and length columns
                one_cont_one_len_df['Contingency'] = self.conts[cont] if self.conts != 'all' else cont
                one_cont_one_len_df['Length'] = length + 1

                # Vertical stack the new dataframe to the sequence rates dataframe
                if seq_rates_df.empty:
                    seq_rates_df = one_cont_one_len_df
                else:
                    seq_rates_df = pd.concat([seq_rates_df, one_cont_one_len_df], axis=0)

        # Studentized Test Statistic
        valid_indices = (seq_rates_df['Group 1 Var'] > 0) & (seq_rates_df['Group 2 Var'] > 0)  # Only calculate when the standard deviation is defined and both are non-zero
        c1c2 = np.zeros(seq_rates_df.shape[0])  # Number of zeroes equal to the number of sequences
        c1c2[valid_indices] = (seq_rates_df['Group 1 Avg'][valid_indices] - seq_rates_df['Group 2 Avg'][valid_indices]) / np.sqrt((seq_rates_df['Group 1 Var'][valid_indices] / seq_rates_df['Group 1 N'][valid_indices]) + (seq_rates_df['Group 2 Var'][valid_indices] / seq_rates_df['Group 2 N'][valid_indices]))
        seq_rates_df['Studentized Test Statistic 1'] = c1c2
        seq_rates_df['Studentized Test Statistic 2'] = -c1c2
        return seq_rates_df
    
    def getStudentizedTestStatsPD(self, groups: list[np.array]):
        """
        Given a list of groups of sequences, returns the sequence rates for each group.
        The columns are:
            Group 1 Averages ... Group n Averages, Original Sequence Number, Length, Contingency
        The rows are sequences, regardless of length and contingency.
        """
        seq_num_col_name = "Original Seq No."
        seq_rates_df = pd.DataFrame()
        lengths, conts = self.all_seqcnts_matrix.shape
        for cont in np.arange(conts):
            for length in np.arange(lengths):
                seq_cnts = pd.DataFrame(self.all_seqcnts_matrix[length][cont])
                one_cont_one_len_df = pd.DataFrame()
                for i, group in enumerate(groups):
                    seq_cnts_g = seq_cnts.iloc[group]  # Keep only the animals (rows) that are in the group
                    seq_cnts_g = seq_cnts_g.loc[~(seq_cnts_g == self.CONSTANTS['NaN']).any(axis=1)]  # Remove animals (rows) with -1 (NaN) in the sequence counts
                    seq_cnts_g = seq_cnts_g.T  # Transpose so that sequences are rows and animals are columns
                    variance = seq_cnts_g.var(axis=1)  # Calculate the variance of the sequence counts (each row) for each sequence
                    n = seq_cnts_g.shape[1]  # Number of animals in the group
                    seq_cnts_g = seq_cnts_g.mean(axis=1)  # Average the sequence counts (each row) for each sequence
                    seq_cnts_g = seq_cnts_g.reset_index()
                    seq_cnts_g.columns = [seq_num_col_name, f'Group {i+1} Avg']
                    seq_cnts_g[f'Group {i+1} Var'] = variance
                    seq_cnts_g[f'Group {i+1} N'] = n # Number of animals in the group

                    # Combine the dataframes from each group
                    if one_cont_one_len_df.empty:
                        one_cont_one_len_df = seq_cnts_g
                    else:
                        one_cont_one_len_df = pd.merge(one_cont_one_len_df, seq_cnts_g, on=seq_num_col_name)

                # Vertical stack the new dataframe to the sequence rates dataframe
                if seq_rates_df.empty:
                    seq_rates_df = one_cont_one_len_df
                else:
                    seq_rates_df = pd.concat([seq_rates_df, one_cont_one_len_df], axis=0)

        # Studentized Test Statistic
        valid_indices = (seq_rates_df['Group 1 Var'] > 0) & (seq_rates_df['Group 2 Var'] > 0)  # Only calculate when the standard deviation is defined and both are non-zero
        c1c2 = np.zeros(seq_rates_df.shape[0])  # Number of zeroes equal to the number of sequences
        c1c2[valid_indices] = (seq_rates_df['Group 1 Avg'][valid_indices] - seq_rates_df['Group 2 Avg'][valid_indices]) / np.sqrt((seq_rates_df['Group 1 Var'][valid_indices] / seq_rates_df['Group 1 N'][valid_indices]) + (seq_rates_df['Group 2 Var'][valid_indices] / seq_rates_df['Group 2 N'][valid_indices]))
        result = np.repeat(c1c2, 2)
        result[1::2] *= -1
        return result

    def getStudentizedTestStats(self, groups: list[np.array], abbrev=False):
        """
        Generates one row of the resampled matrix.
        Calculates the studentized test statistics for each sequence.
        """
        sts = None
        lengths, conts = self.all_seqcnts_matrix.shape
        for cont in np.arange(conts):
            for length in np.arange(lengths):
                seq_cnts = self.all_seqcnts_matrix[length][cont]
                # Always 2 groups
                # Group 1
                seq_cnts_g1 = seq_cnts[groups[0]]
                seq_cnts_g1 = seq_cnts_g1[~(seq_cnts_g1 == self.CONSTANTS['NaN']).any(axis=1)]
                n1 = seq_cnts_g1.shape[0]
                var1 = seq_cnts_g1.var(axis=0)
                mean1 = seq_cnts_g1.mean(axis=0)
                # Group 2
                seq_cnts_g2 = seq_cnts[groups[1]]
                seq_cnts_g2 = seq_cnts_g2[~(seq_cnts_g2 == self.CONSTANTS['NaN']).any(axis=1)]
                n2 = seq_cnts_g2.shape[0]
                var2 = seq_cnts_g2.var(axis=0)
                mean2 = seq_cnts_g2.mean(axis=0)
                # Studentized Test Statistic
                valid_indices = (var1 > 0) & (var2 > 0)  # Only calculate when the standard deviation is defined and both are non-zero
                c1c2 = np.zeros(len(mean1))
                c1c2[valid_indices] = (mean1[valid_indices] - mean2[valid_indices]) / np.sqrt((var1[valid_indices] / n1) + (var2[valid_indices] / n2))
                

                result = np.repeat(c1c2, 2)
                result[1::2] *= -1
                if sts is None:
                    sts = result
                else:
                    sts = np.hstack((sts, result))
        if abbrev:
            paired_result = [(idx, val) for idx, val in enumerate(sts) if val > 0]
            return paired_result
        else:
            return sts
    
    def resample(self, id=0):
        """Performs one resampling of the groups."""
        # Resample the groups
        resampled_groups = self.resampleGroups(id)
        # Calculate the studentized test statistics for the resampled groups
        return self.getStudentizedTestStatsPD(resampled_groups)
    
    
    def generateResampledMatrixParallel(self, num_resamples=10000):
        """
        Gets the studentized test statistics for the original groups, 
        then resamples the groups and calculates the studentized test statistics for each sequence.
        """
        # First do it for the original groups
        reference_studentized_test_stats = self.getStudentizedTestStatsPD(self.orig_groups)

        # Create a pool of worker processes
        pool = multiprocessing.Pool()
        # Run self.sample() and append the result to resampled_matrix at the specified index
        resampled_matrix = np.empty((num_resamples+1, self.totalSeqs() * 2))
        resampled_matrix[0] = reference_studentized_test_stats
        results = pool.map(self.resample, np.arange(1, num_resamples+1))
        for i, result in enumerate(results):
            resampled_matrix[i+1] = result
            results[i] = None  # Remove the row from the results to free up memory
        # Close the pool and wait for all processes to finish
        pool.close()
        pool.join()

        return resampled_matrix
    
    def generateResampledMatrixParallelAbbrev(self, num_resamples=10000):
        """
        Gets the studentized test statistics for the original groups, 
        then resamples the groups and calculates the studentized test statistics for each sequence.
        """
        resampled_matrix = []
        # First do it for the original groups
        studentized_test_stats = self.getStudentizedTestStats(self.orig_groups)

        # Set the first row of the resampled_matrix
        resampled_matrix.append(studentized_test_stats)

        # Create a pool of worker processes
        pool = multiprocessing.Pool()
        # Run self.sample() and append the result to resampled_matrix
        resampled_matrix.extend(pool.map(self.resample, range(1, num_resamples+1)))
        # Close the pool and wait for all processes to finish
        pool.close()
        pool.join()

        return resampled_matrix

        


    def writeSequenceRatesFile(self, seq_rates_df: pd.DataFrame):
        # Write the sequence rates matrix to a file
        # Get the matrix without its column header
        mat = seq_rates_df.to_numpy()
        seq_rates_f = CBASFile("seqRates", mat)
        seq_rates_f.saveFile(self.FILES['OUTPUT'])
        
    def writeResampledMatrix(self, resampled_matrix, filename='resampled_matrix'):
        # Write the resampled matrix to a file
        # np.savetxt(os.path.join(self.FILES['OUTPUT'],
        #                         f'{filename}.csv'), 
        #             resampled_matrix, delimiter=',')

        # Pickle the resampled matrix
        cbas_file = CBASFile(filename, resampled_matrix)
        cbas_file.saveFile(self.FILES['OUTPUT'], use_sparsity_csr=True, dtype=float)
        # FileUtils.pickleObj(resampled_matrix, os.path.join(self.FILES['OUTPUT'], f'{filename}.pkl'))
