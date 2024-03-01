from sequences import SequencesProcessor
from files import FileManager
from settings import Settings

import sys
import os
import time
import numpy as np

import datetime

def format_time(seconds):
    duration = datetime.timedelta(seconds=seconds)
    if duration < datetime.timedelta(milliseconds=1):
        return "{:.2f} microseconds".format(duration.microseconds)
    elif duration < datetime.timedelta(seconds=1):
        return "{:.2f} milliseconds".format(duration.microseconds / 1000)
    elif duration < datetime.timedelta(minutes=1):
        return "{:.2f} seconds".format(duration.total_seconds())
    elif duration < datetime.timedelta(hours=1):
        return "{:.2f} minutes".format(duration.total_seconds() / 60)
    elif duration < datetime.timedelta(days=1):
        return "{:.2f} hours".format(duration.total_seconds() / 3600)
    else:
        return "{:.2f} days".format(duration.total_seconds() / 86400)

if __name__ == "__main__":
    start = time.time()

    section_start = time.time()
    print("Starting PythonCBAS Engine...")
    print("Getting settings...")
    settings = Settings()
    settings.setCriterion({'ORDER': 0, 'NUMBER': float('inf'), 'INCLUDE_FAILED': True, 'ALLOW_REDEMPTION': False})

    FILES = settings.getFiles()
    ANIMAL_FILE_FORMAT = settings.getAnimalFileFormat()
    LANGUAGE = settings.getLanguage()
    CRITERION = settings.getCriterion()
    print("Settings retrieved. Time taken: ", format_time(time.time() - section_start))

    section_start = time.time()
    print("Setting up files...")
    fileManager = FileManager(FILES)
    fileManager.setupFiles()
    print("Files set up. Time taken: ", format_time(time.time() - section_start))

    section_start = time.time()
    print("Processing sequences...")
    sequencesProcessor = SequencesProcessor(FILES, ANIMAL_FILE_FORMAT, LANGUAGE, CRITERION)
    sequencesProcessor.processAllAnimals()
    print("Sequences processed. Time taken: ", format_time(time.time() - section_start))

    section_start = time.time()
    print("Generating sequence files...")
    sequencesProcessor.generateSequenceFiles()
    print("Sequence files generated. Time taken: ", format_time(time.time() - section_start))

    section_start = time.time()
    print("Finding criterion trial for just cont 0 len 1...")
    sequencesProcessor.findCriterionTrial(0, 2)
    print(f"Total Time: {format_time(time.time() - start)}")


    sys.exit()
