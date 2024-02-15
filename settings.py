import os


class Settings:
    def __init__(self):
        self.language = {
            'NUM_CHOICES': 6,
            'STRADDLE_SESSIONS': False,
            'NUM_MODIFIERS': 1,
            'MODIFIER_RANGES': [6],
            'MAX_SEQUENCE_LENGTH': 6,
        }

        self.files = {
            'DATA': 'scn2aDataSwapped/',
            'OUTPUT': 'output/',
            'METADATA': 'metadata/',
            'EXPECTED_OUTPUT': 'expected_output_scn2a/',
            'COHORTS_FILE': os.path.join('metadata', 'cohorts.txt'),
            'ANIMALS_FILE': os.path.join('metadata', 'animals.txt'),
            'INFO_FILE': 'anInfo.txt',
        }
        
        self.animal_file_format = {
            'SESSION_NO_COL': 0,
            'CHOICE_COL': 1,
            'CONTINGENCY_COL': 2,
            'MODIFIER_COL': 3,
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

    def getCatalogue(self):
        return self.catalogue   
    
    def setLanguage(num_choices, modifier_ranges):
        """Sets the language for the experiment. The language is defined by the number of choices and the number of modifiers."""
        NUM_CHOICES = num_choices
        NUM_MODIFIERS = len(modifier_ranges)
        MODIFIER_RANGES = modifier_ranges




