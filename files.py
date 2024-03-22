import os
from pathlib import Path
import re
import numpy as np
import time
import pickle
from sys import getsizeof






class Header:
    """
    Represents either a column or row header in a table-like structure.
    Used to identify the type of data in the header, which dictates what actions can be performed on it.
    """
    cbas_types = ['sequence', 'animal', 'cohort', 'trial']

    def __init__(self, axis, header_map: dict=None, header_type=None, header_size=None, custom_types: list[str]=None):
        # Process and type check the axis information
        if (type(axis) == int) and (axis in [0, 1]):
            self.axis = axis
        elif (type(axis) == str):
            if axis.lower() in 'row':
                self.axis = 0
            elif axis.lower() in 'column':
                self.axis = 1
            else:
                raise ValueError("Axis must be either 0, 1, 'row', or 'column'")
        
        self.types = Header.cbas_types
        if custom_types is not None:
            self.types += custom_types

        # Process and type check the header information. 
        # Used to determine whether this header is one type across the entire header or multiple individual types.
        if header_map is None:
            if (header_type is not None) and (header_size is not None):
                raise ValueError("If header_map is None, header_type and header_size must be specified")
            if header_type not in self.types:
                raise ValueError(f"Invalid header type {header_type}")
            self.header_type = header_type
            self.header_labels = [i for i in np.arange(header_size)]
        else:
            if not all(type(t) == str for t in header_map.values()):
                raise ValueError("All values in header_map must be strings")
            if all(t.lower() in self.types for t in header_map.values()):
                self.header_labels = header_map.keys()
                self.header_types = header_map.values()

    def isOneHeaderType(self):
        """If the header is one type across the entire header, return True. Otherwise, return False."""
        return hasattr(self, 'header_type')
    
    def isMultiHeaderTypes(self):
        """If the header is multiple types for each individual header, return True. Otherwise, return False."""
        return hasattr(self, 'header_types')

    def getHeaderLabels(self):
        return self.header_labels
    
    def getHeaderType(self, idx: int):
        if self.isOneHeaderType():
            return self.header_type
        elif self.isMultiHeaderTypes():
            return self.header_types[idx]
        


class CBASFile:
    def __init__(self, name, type, data, hheader: Header, vheader: Header):
        self.name = name
        self.type = type
        self.data = data
        assert hheader.axis == 1  # Horizontal headers describe columns
        assert vheader.axis == 0  # Vertical headers describe rows

    def export(self, filepath, type="csv"):
        """Exports the file to the given type"""
        if type == "csv":
            # TODO: Type checking
            FileManager.writeMatrix(filepath, self.data)

    def __eq__(self, other):
        pass

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

    def getMatrix(file, delimiter=','):
        """Takes a text file and returns a numpy matrix"""
        # Force it to be 2D even if there's only one row
        return np.atleast_2d(np.genfromtxt(file, delimiter=delimiter, dtype=int))
    
    def writeMatrix(file, mat):
        """Writes a numpy matrix to a text file. Don't write an extra line"""
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
                animal_info_matrix = FileManager.getMatrix(info_file, delimiter='\t')
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
        
    def writeCBASFile(filepath, fileobj: CBASFile):
        """Writes a CBASFile object to a file"""
        FileManager.pickle_obj(fileobj, filepath)

    def readCBASFile(filepath) -> CBASFile:
        """Reads a CBASFile object from a file"""
        return FileManager.unpickle_obj(filepath)

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
        # Clear output directory
        if not os.path.exists(self.FILES['OUTPUT']):
            os.rmdir(self.FILES['OUTPUT'])
        if not os.path.exists(self.FILES['METADATA']):
            os.rmdir(self.FILES['METADATA'])

        Path(self.FILES['METADATA']).mkdir(parents=True, exist_ok=True)
        self.clearMetadata()

        Path(self.FILES['OUTPUT']).mkdir(parents=True, exist_ok=True)
        Path(self.FILES['ALLSEQDIR']).mkdir(parents=True, exist_ok=True)
        Path(self.FILES['ALLSEQALLANDIR']).mkdir(parents=True, exist_ok=True)
        Path(self.FILES['SEQCNTSDIR']).mkdir(parents=True, exist_ok=True)
        


        cohort_dict = self.buildCohortInfo()
        self.buildAnimalInfo(cohort_dict)

    def compareFiles(file1, file2):
        """Compares two files and returns True if they are the same, False otherwise"""
        with open(file1, 'r') as f1, open(file2, 'r') as f2:
            return f1.read() == f2.read()
        




