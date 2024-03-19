import unittest
from files import FileManager
from settings import Settings
import re
import os
import pandas as pd
import numpy as np
from random import randint, seed


class CheckSequenceFiles(unittest.TestCase):

    def setUp(self):
        settings = Settings()
        settings.setCriterion({'ORDER': 4, 'NUMBER': 100, 'INCLUDE_FAILED': True, 'ALLOW_REDEMPTION': True})
        self.files = settings.getFiles()
        self.expected_dir = os.path.join(self.files['EXPECTED_OUTPUT'], "zerothInf")
        self.output_dir = self.files['OUTPUT']
    
    def test_allSeq_files(self):
        """
        Compare the size of all files from the expected directory whose name starts with 'allSeq_'
        """
        expected_files = [f for f in os.listdir(self.expected_dir) if re.match(r'allSeq_', f) and (not f.startswith('.')) and not f.startswith('allSeq_0')]
        output_files = [f for f in os.listdir(self.output_dir) if re.match(r'allSeq_', f) and (not f.startswith('.')) and not f.startswith('allSeq_0')]
        for f in expected_files:
            self.assertTrue(f in output_files, f"File {f} not found in output directory.")
            # Expected output needs to be transposed first
            expected_matrix = np.loadtxt(os.path.join(self.expected_dir, f), delimiter='\t', dtype=int).T
            output_matrix = np.loadtxt(os.path.join(self.output_dir, f), delimiter=',', dtype=int)
            # Check if size matches
            self.assertEqual(expected_matrix.shape, output_matrix.shape, f"File {f} is not the same size in the output directory as in the expected directory.")

    def test_allSeqAllAn_files(self):
        """
        Compare the contents of all files from the expected directory whose name starts with 'allSeqAllAn_'
        """
        expected_files = [f for f in os.listdir(self.expected_dir) if re.match(r'allSeqAllAn_', f) and (not f.startswith('.')) and not f.startswith('allSeqAllAn_0')]
        output_files = [f for f in os.listdir(self.output_dir) if re.match(r'allSeqAllAn_', f) and (not f.startswith('.')) and not f.startswith('allSeqAllAn_0')]
        for f in expected_files:
            self.assertTrue(f in output_files, f"File {f} not found in output directory.")
            # Get the second column of each matrix
            expected = list(np.loadtxt(os.path.join(self.expected_dir, f), delimiter='\t', dtype=int)[:, 1])
            output = list(np.loadtxt(os.path.join(self.output_dir, f), delimiter=',', dtype=int)[:, 1])

            # Sort the lists and see if they match
            expected.sort()
            output.sort()
            self.assertEqual(expected, output, f"File {f} is not the same in the output directory as in the expected directory.")

    def test_criterion_file(self):
        """
        Compare the criterion file generated for Order 4 Number 100 Include Failed True Allow Redemption True
        File names are the same
        """
        # self.assertTrue(FileManager.compareFiles(os.path.join(self.expected_dir, 'criterionMatrix_0_inf_True_True.txt'), os.path.join(self.output_dir, 'criterionMatrix_0_inf_True_True.txt')), "Criterion file is not the same in the output directory as in the expected directory.")
        expected_matrix = np.loadtxt(os.path.join(self.expected_dir, 'criterionMatrix_4_100_True_True.csv'), delimiter=',')
        output_matrix = np.loadtxt(os.path.join(self.output_dir, 'criterionMatrix_4_100_True_True.txt'), delimiter=',')[:,1:]
        self.assertEqual(expected_matrix.shape, output_matrix.shape, "Criterion file is not the same size in the output directory as in the expected directory.")
        # For every row in the expected, ensure it's also in the output
        correct = 0
        total = expected_matrix.shape[0]
        for i in range(expected_matrix.shape[0]):
            # If the row exists...
            if np.any(np.all(expected_matrix[i] == output_matrix, axis=1)):
                correct += 1
        print(f"Correct: {correct}, Total: {total}" )
        self.assertEqual(correct, total, f"Criterion file is not the same in the output directory as in the expected directory.")



    def test_seqCnts_file(self):
        """
        Compare the seqCnts file generated
        File names are the same
        """
        # Check if the sizes are the same
        expected_files = [f for f in os.listdir(self.expected_dir) if re.match(r'seqCnts_', f) and not f.startswith('.')]
        output_files = [f for f in os.listdir(self.output_dir) if re.match(r'seqCnts_', f) and not f.startswith('.')]
        total = 0
        correct = 0
        for f in expected_files:
            self.assertTrue(f in output_files, f"File {f} not found in output directory.")
            # Read the files using pandas
            expected = pd.read_csv(os.path.join(self.expected_dir, f), delimiter=',', dtype=int)
            output = pd.read_csv(os.path.join(self.output_dir, f), delimiter=',', dtype=int)
            # Check if the sizes are the same
            if expected.shape == output.shape:
                correct += 1
            else:
                print(f"Size difference for file {f}: {expected.shape} vs {output.shape}")
            total += 1
        self.assertTrue(correct == total, f"Only {correct}/{total} files have the same size in the output directory as in the expected directory.")

class CheckSequenceFilesDeep(unittest.TestCase):
    
        def setUp(self):
            settings = Settings()
            self.files = settings.getFiles()
            self.expected_dir = os.path.join(self.files['EXPECTED_OUTPUT'], "zerothInf")
            self.output_dir = self.files['OUTPUT']

        def translateExpectedToOutputSeqNum(self, cont, length, expected_seq_num):
            """
            Takes a sequence number in the expected set and returns the corresponding sequence number in the output set
            """
            expected_allSeq = np.atleast_2d(np.loadtxt(os.path.join(self.expected_dir, f'allSeq_{cont}_{length}.txt'), delimiter='\t', dtype=int)).T
            expected_sequence = tuple(expected_allSeq[expected_seq_num])
            
            output_allSeq = np.atleast_2d(np.loadtxt(os.path.join(self.output_dir, f'allSeq_{cont}_{length}.txt'), delimiter=',', dtype=int))
            if length == 1:
                output_allSeq = output_allSeq.T
            output_sequence_idx = np.where((output_allSeq == expected_sequence).all(axis=1))[0][0]
            return output_sequence_idx

        def natural_sort(self, l):
            convert = lambda text: int(text) if text.isdigit() else text
            alphanum_key = lambda key: [convert(c) for c in re.split('([0-9]+)', key)]
            return sorted(l, key=alphanum_key)

        def test_translate(self):
            print(self.translateExpectedToOutputSeqNum(0, 1, 0))
        
        def test_seqCnts_deep(self, percent=0.3):
            """
            Compare the contents of the sequence counts matrices by choosing 30% of the sequences at random
            """
            seed(925)
            expected_files = [f for f in os.listdir(self.expected_dir) if re.match(r'seqCnts_', f)]
            expected_files = self.natural_sort(expected_files)
            output_files = [f for f in os.listdir(self.output_dir) if re.match(r'seqCnts_', f)]
            overall_good = 0
            overall_checked = 0
            for f in expected_files:
                self.assertTrue(f in output_files, f"File {f} not found in output directory.")
                # Cont is the number after the first _
                cont = int(f.split('_')[1])
                # Length is the number after the second _
                length = int(f.split('_')[2].split('.')[0])
                # Read the files using numpy. Set blank values to -1
                expected = np.loadtxt(os.path.join(self.expected_dir, f), delimiter=',', dtype=str)
                expected[expected == ''] = -1
                # Convert the expected back into integers
                expected = expected.astype(int)
                output = np.loadtxt(os.path.join(self.output_dir, f), delimiter=',', dtype=int)

                good = 0
                # Choose percent% of the columns at random
                for _ in range(int(percent * expected.shape[1])):
                    col_expected = randint(0, expected.shape[1] - 1)
                    col_output = self.translateExpectedToOutputSeqNum(cont, length, col_expected)
                    
                    # Sort the columns and check if they match
                    expected_col = list(expected[:, col_expected])
                    output_col = list(output[:, col_output])
                    
                    expected_sum = np.sum(expected_col)
                    output_sum = np.sum(output_col)
                    expected_col.sort()
                    output_col.sort()
                    # percent_match = len(set(expected_col) & set(output_col)) / len(set(expected_col))
                    
                    if expected_sum == output_sum:
                        good += 1
                # print(f"For file {f}, number of sum matches: {good}/{int(percent * expected.shape[1])} ({np.round(100 * good/int(percent * expected.shape[1]), 2)}%)")
                total_cols_checked = int(expected.shape[1] * percent)
                columns_checked_str = f"{total_cols_checked}/{expected.shape[1]} of the columns checked."
                num_tabs = 2 if len(columns_checked_str) < 29 else 1
                tabs = '\t' * (num_tabs + 1)
                print(f"For file {f}:  {columns_checked_str}{tabs}Of these, {good}/{total_cols_checked} matched ({np.round(100 * good/total_cols_checked, 2)}%).")

                overall_good += good
                overall_checked += total_cols_checked
            print("-" * 100)
            print(f"Overall, {overall_good}/{overall_checked} matched ({np.round(100 * overall_good/overall_checked, 2)}%).")
            self.assertEqual(overall_good, overall_checked, "Not all sequences matched.")

        def test_seqCnts_deep_all(self):
            """Same as the test_seqCnts_deep, but for all sequences"""
            expected_files = [f for f in os.listdir(self.expected_dir) if re.match(r'seqCnts_', f) and (not f.startswith('.')) and not f.startswith('seqCnts_0')]
            expected_files = self.natural_sort(expected_files)
            output_files = [f for f in os.listdir(self.output_dir) if re.match(r'seqCnts_', f) and (not f.startswith('.')) and not f.startswith('seqCnts_0')]
            overall_good = 0
            overall_checked = 0
            for f in expected_files:
                self.assertTrue(f in output_files, f"File {f} not found in output directory.")
                # Cont is the number after the first _
                cont = int(f.split('_')[1])
                # Length is the number after the second _
                length = int(f.split('_')[2].split('.')[0])
                # Read the files using numpy. Set blank values to -1
                expected = np.loadtxt(os.path.join(self.expected_dir, f), delimiter='\t', dtype=str)
                expected[expected == ''] = -1
                # Convert the expected back into integers
                expected = expected.astype(int)
                output = np.loadtxt(os.path.join(self.output_dir, f), delimiter=',', dtype=int)

                good = 0
                expected_sum, output_sum = 0, 0
                for col in range(expected.shape[1]):
                    col_output = self.translateExpectedToOutputSeqNum(cont, length, col)
                    # Sort the columns and check if they match
                    expected_col = list(expected[:, col])
                    output_col = list(output[:, col_output])
                    expected_col.sort()
                    output_col.sort()
                    expected_sum += np.sum(expected_col)
                    output_sum += np.sum(output_col)
                    if expected_col == output_col:
                        good += 1
                print(f"For file {f}, number of matches: {good}/{expected.shape[1]} ({np.round(100 * good/expected.shape[1], 2)}%)")
                overall_good += good

                overall_checked += expected.shape[1]
            print("-" * 100)
            print(f"Overall, {overall_good}/{overall_checked} matched ({np.round(100 * overall_good/overall_checked, 2)}%).")
            self.assertEqual(overall_good, overall_checked, "Not all sequences matched.")



if __name__ == "__main__":
    # Run only the tests in the CheckSequenceFilesDeep class
    suite = unittest.TestSuite()
    # suite.addTest(CheckSequenceFiles('test_criterion_file'))
    # suite.addTest(CheckSequenceFilesDeep('test_seqCnts_deep'))

    suite.addTest(CheckSequenceFiles('test_allSeq_files'))
    suite.addTest(CheckSequenceFiles('test_allSeqAllAn_files'))
    suite.addTest(CheckSequenceFilesDeep('test_seqCnts_deep_all'))

    # suite.addTest(CheckSequenceFilesDeep('test_translate'))
    runner = unittest.TextTestRunner()
    runner.run(suite)


    
