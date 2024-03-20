import os
import pandas as pd

class Settings:
    def __init__(self):
        self.language = {
            'NUM_CHOICES': 6,
            'STRADDLE_SESSIONS': False,
            'NUM_MODIFIERS': 1,
            'MODIFIER_RANGES': [6],
            'MAX_SEQUENCE_LENGTH': 6,
            'NUM_CONTINGENCIES': 7,
        }

        self.criterion = {
            'ORDER': 0,
            'NUMBER': float('inf'),
            'INCLUDE_FAILED': True,  # Special case: If subject does not get to the criterion (e.g., want 200th trial but subject only completes 30)
            'ALLOW_REDEMPTION': True,  # Do we exclude for every (subsequent) contingency (False), 
                                        # or do we exclude it until it reaches the criterion again (True)?
        }

        self.files = {
            'DATA': 'Data/scn2aDataSwapped/',
            'OUTPUT': 'output/',
            'METADATA': 'metadata/',
            'EXPECTED_OUTPUT': 'expected_output_scn2a/',
            'COHORTS_FILE': os.path.join('metadata', 'cohorts.txt'),
            'ANIMALS_FILE': os.path.join('metadata', 'animals.txt'),
            'INFO_FILE': 'anInfo.txt',
        }

        self.animal_info_format = {
            'ANKEY': 0,
            'GENOTYPE': 1,
            'SEX': 2,
            'LESION': 3,
            'IMPLANT': 4,
        }
        
        self.animal_file_format = {
            'SESSION_NO_COL': 0,
            'CHOICE_COL': 1,
            'CONTINGENCY_COL': 2,
            'MODIFIER_COL': 3,
        }

        self.constants = {
            'NaN': -1,
            'inf': -2
        }

        # Catalogue for the possible values of each attribute, in numerical order (0, 1, 2, ...)
        # For example, since while making the data, we set 0 to be male and 1 to be female, and it
        # appears this way in anInfo.txt, we have under SEX, 'Male' in index 0, and 'Female' in index 1.
        self.catalogue = {
            'GENOTYPE': ['WT', 'scn2a heterozygous'],
            'SEX': ['Male', 'Female'],
            'LESION': ['Unlesioned', 'Hippocampus', 'DMS', 'DREADD'],
            'IMPLANT': ['Unimplanted', 'Implanted']
        }

    def getLanguage(self):
        return self.language

    def getFiles(self):
        return self.files

    def getAnimalFileFormat(self):
        return self.animal_file_format
    
    def getCriterion(self):
        return self.criterion
    
    def getConstants(self):
        return self.constants

    def getCatalogue(self):
        return self.catalogue 
    
    def getAnInfoCol(self, col_name):
        return self.animal_info_format[col_name]


    
    def setLanguage(num_choices, modifier_ranges):
        """Sets the language for the experiment. The language is defined by the number of choices and the number of modifiers."""
        NUM_CHOICES = num_choices
        NUM_MODIFIERS = len(modifier_ranges)
        MODIFIER_RANGES = modifier_ranges

    def setCriterion(self, attr_dict: dict):
        """Sets the criterion for the experiment. The criterion is defined by the order, number, whether to include failed trials, and whether to allow redemption."""
        assert attr_dict.keys() == self.criterion.keys()
        self.criterion = attr_dict

    def assignGroups(self, groups: list[dict]):
        """Assigns animals to groups based on the filters provided."""
        animal_matrix = pd.read_csv(self.files['ANIMALS_FILE'], header=None)
        an_nums = []
        for group in groups:
            animal_matrix_copy = animal_matrix.copy()
            # Each element in the group is a filter
            for col_name, value in group.items():
                col_num = self.animal_info_format[col_name] + 2  # +2 because the first two columns are animal number and cohort number
                animal_matrix_copy = animal_matrix_copy[animal_matrix_copy[col_num] == value]
            an_nums.append(list(animal_matrix_copy[0]))
        return an_nums

            

    # def setCriterion(self, order: int, number: int, include_failed: bool, allow_redemption: bool):
    #     """Sets the criterion for the experiment. The criterion is defined by the order, number, whether to include failed trials, and whether to allow redemption."""
    #     self.criterion['ORDER'] = order
    #     self.criterion['NUMBER'] = number
    #     self.criterion['INCLUDE_FAILED'] = include_failed
    #     self.criterion['ALLOW_REDEMPTION'] = allow_redemption




