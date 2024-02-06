from sequences import SequencesProcessor
from files import FileManager
from settings import Settings

import sys
import os
import time



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

    sequencesProcessor = SequencesProcessor(ANIMAL_FILE_FORMAT, LANGUAGE)
    
    print(f"Total Time: {time.time() - start}")
    sys.exit()

    mat = getMatrix(os.path.join(FILES['DATA'], 'scn2aCoh1', 'anData0.txt'))
    mat = collapseModifiers(mat)
    # print(splitContingency(mat)[0])
    print(mat)
