import os
import re
import numpy as np
import time
import pickle
from sys import getsizeof






### SETUP METADATA ###
class FileManager:
    def __init__(self, FILES):
        self.FILES = FILES

    def natural_sort(l):
        convert = lambda text: int(text) if text.isdigit() else text
        alphanum_key = lambda key: [convert(c) for c in re.split('([0-9]+)', key)]
        return sorted(l, key=alphanum_key)

    def buildCohortInfo(self):
        """
        Goes through each cohort and assigns a number to each cohort. Write it to the cohorts file in comma-separated format:
            cohort_name, cohort_number

        Simultaneously builds a dictionary of the cohort names and their corresponding numbers and returns it.
        We build the dictionary simultaneously for consistency, so that the file and the dictionary are in sync.
        """
        # Get the list of all the cohort folder and sort by natural order
        cohort_folders = FileManager.natural_sort([name for name in os.listdir(self.FILES['DATA']) if os.path.isdir(os.path.join(self.FILES['DATA'], name))])
        # Write the cohort names and their corresponding numbers to the cohorts file
        with open(self.FILES['COHORTS_FILE'], 'w') as f:
            for i, cohort in enumerate(cohort_folders):
                f.write(f"{cohort},{i}\n")
        return {cohort: i for i, cohort in enumerate(cohort_folders)}

    def getMatrix(file):
        """Takes a text file and returns a numpy matrix"""
        # Force it to be 2D even if there's only one row
        return np.atleast_2d(np.genfromtxt(file, delimiter=',', dtype=int))
    
    def writeMatrix(file, mat):
        """Writes a numpy matrix to a text file"""
        np.savetxt(file, mat, delimiter=',', fmt='%d')

    def buildAnimalInfo(self, cohort_dict):
        """
        Goes through the cohorts in order of the cohort_dict and processes the animal information for each cohort.
        Assigns each animal a number and writes it to the animals file in comma-separated format:
            animal number, cohort number, <all info in the info file>

        Note:
        The order of animals in animals.txt is the same as the order of animals in all_paths.pkl
        """
        animal_num = 0
        all_paths = []
        with open(self.FILES['ANIMALS_FILE'], 'a') as f:
            for cohort_name, cohort_num in cohort_dict.items():
                cohort_folder = os.path.join(self.FILES['DATA'], cohort_name)
                info_file = os.path.join(cohort_folder, self.FILES['INFO_FILE'])
                animal_files = FileManager.natural_sort([name for name in os.listdir(cohort_folder) if os.path.isfile(os.path.join(cohort_folder, name)) 
                                                         and name != self.FILES['INFO_FILE'] 
                                                         and not name.startswith('.')])
                all_paths += [os.path.join(cohort_folder, file) for file in animal_files]
                animal_info_matrix = FileManager.getMatrix(info_file)
                # Get rid of hidden files
                animal_files = [file for file in animal_files if not file.startswith('.')]
                assert len(animal_files) == len(animal_info_matrix)
                for i, animal_file in enumerate(animal_files):
                    animal_info = animal_info_matrix[i]
                    f.write(f"{animal_num},{cohort_num},{','.join([str(int(x)) for x in animal_info])}\n")
                    animal_num += 1
        FileManager.pickle_obj(all_paths, os.path.join(self.FILES['METADATA'], 'all_paths.pkl'))

    def pickle_obj(obj, file):
        with open(file, 'wb') as f:
            pickle.dump(obj, f)

    def unpickle_obj(file):
        with open(file, 'rb') as f:
            return pickle.load(f)

    def clearMetadata(self):
        """
        Clears the metadata files.
        """
        with open(self.FILES['COHORTS_FILE'], 'w') as f:
            f.write('')
        with open(self.FILES['ANIMALS_FILE'], 'w') as f:
            f.write('')

    def setupFiles(self):
        """
        MAIN FUNCTION FOR THIS MODULE
        Sets up the output and metadata folders and ensures they all exist.
        Then processes the cohort and animal information and writes it to the metadata files.
        """
        if not os.path.exists(self.FILES['OUTPUT']):
            os.makedirs(self.FILES['OUTPUT'])
        if not os.path.exists(self.FILES['METADATA']):
            os.makedirs(self.FILES['METADATA'])

        self.clearMetadata()

        cohort_dict = self.buildCohortInfo()
        self.buildAnimalInfo(cohort_dict)

