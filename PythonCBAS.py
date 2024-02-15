from sequences import SequencesProcessor
from files import FileManager
from settings import Settings

import sys
import os
import time
import numpy as np



if __name__ == "__main__":
    start = time.time()

    print("Starting PythonCBAS Engine...")
    print("Getting settings...")
    settings = Settings()
    FILES = settings.getFiles()
    ANIMAL_FILE_FORMAT = settings.getAnimalFileFormat()
    LANGUAGE = settings.getLanguage()
    print("Settings retrieved. Time taken: ", time.time() - start)

    print("Setting up files...")
    fileManager = FileManager(FILES)
    fileManager.setupFiles()
    print("Files set up. Time taken: ", time.time() - start)

    sequencesProcessor = SequencesProcessor(FILES, ANIMAL_FILE_FORMAT, LANGUAGE)
    sequencesProcessor.generateSequenceFiles()

    print(f"Total Time: {time.time() - start}")
    sys.exit()

    sequencesProcessor = SequencesProcessor(FILES, ANIMAL_FILE_FORMAT, LANGUAGE)
    mat = sequencesProcessor.getMatrix(os.path.join(FILES['DATA'], 'scn2aCoh1', 'anData0.txt'))
    mat = sequencesProcessor.collapseModifiers(mat)
    # FileManager.writeMatrix(os.path.join(FILES['OUTPUT'], 'anData0_collapsed.txt'), mat)
    # print(mat[(mat[:, ANIMAL_FILE_FORMAT['CHOICE_COL']] == 12) & (mat[:, ANIMAL_FILE_FORMAT['CONTINGENCY_COL']] == 0)])
    mats = sequencesProcessor.splitContingency(mat)
    sequencesProcessor.getAllLengthSequences(mats)
    # for _ in np.arange(244):
    #     mat = sequencesProcessor.getMatrix(os.path.join(FILES['DATA'], 'scn2aCoh1', 'anData0.txt'))
    #     mat = sequencesProcessor.collapseModifiers(mat)
    #     mats = sequencesProcessor.splitContingency(mat)
    #     sequencesProcessor.getSequences(mats)
    # sequencesProcessor.generateSequenceFiles()
    
    print(f"Total Time: {time.time() - start}")
    

    mat = getMatrix(os.path.join(FILES['DATA'], 'scn2aCoh1', 'anData0.txt'))
    mat = collapseModifiers(mat)
    # print(splitContingency(mat)[0])
    print(mat)
