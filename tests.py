import unittest
from files import FileManager
from settings import Settings
import re
import os
import pandas as pd
import numpy as np


class CheckSequenceFiles(unittest.TestCase):

    def setUp(self):
        settings = Settings()
        settings.setCriterion({'ORDER': 4, 'NUMBER': 100, 'INCLUDE_FAILED': True, 'ALLOW_REDEMPTION': True})
        self.files = settings.getFiles()
        self.expected_dir = self.files['EXPECTED_OUTPUT']
        self.output_dir = self.files['OUTPUT']
    
    def test_allSeq_files(self):
        """
        Compare the size of all files from the expected directory whose name starts with 'allSeq_'
        """
        expected_files = [f for f in os.listdir(self.expected_dir) if re.match(r'allSeq_', f)]
        output_files = [f for f in os.listdir(self.output_dir) if re.match(r'allSeq_', f)]
        for f in expected_files:
            self.assertTrue(f in output_files, f"File {f} not found in output directory.")
            # Expected output needs to be transposed first
            expected_matrix = np.loadtxt(os.path.join(self.expected_dir, f), delimiter=',').T
            output_matrix = np.loadtxt(os.path.join(self.output_dir, f), delimiter=',')
            # Check if size matches
            self.assertEqual(expected_matrix.shape, output_matrix.shape, f"File {f} is not the same size in the output directory as in the expected directory.")

    def test_allSeqAllAn_files(self):
        """
        Compare the size of all files from the expected directory whose name starts with 'allSeqAllAn_'
        """
        expected_files = [f for f in os.listdir(self.expected_dir) if re.match(r'allSeqAllAn_', f)]
        output_files = [f for f in os.listdir(self.output_dir) if re.match(r'allSeqAllAn_', f)]
        for f in expected_files:
            self.assertTrue(f in output_files, f"File {f} not found in output directory.")
            # Get the second column of each matrix
            expected = list(np.loadtxt(os.path.join(self.expected_dir, f), delimiter=',')[:, 1])
            output = list(np.loadtxt(os.path.join(self.output_dir, f), delimiter=',')[:, 1])

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
        self.assertTrue(FileManager.compareFiles(os.path.join(self.expected_dir, 'criterionMatrix_4_100_True_True.txt'), os.path.join(self.output_dir, 'criterionMatrix_4_100_True_True.txt')), "Criterion file is not the same in the output directory as in the expected directory.")

    def test_seqCnts_file(self):
        """
        Compare the seqCnts file generated
        File names are the same
        """
        # Check if the sizes are the same
        expected_files = [f for f in os.listdir(self.expected_dir) if re.match(r'seqCnts_', f)]
        output_files = [f for f in os.listdir(self.output_dir) if re.match(r'seqCnts_', f)]
        for f in expected_files:
            self.assertTrue(f in output_files, f"File {f} not found in output directory.")
            # Read the files using pandas
            expected = pd.read_csv(os.path.join(self.expected_dir, f), delimiter=',')
            output = pd.read_csv(os.path.join(self.output_dir, f), delimiter=',')
            # Check if the sizes are the same
            self.assertEqual(expected.shape, output.shape, f"File {f} is not the same size in the output directory as in the expected directory.")

if __name__ == "__main__":
    # Run tests from CheckSequenceFiles
    unittest.main()
    
