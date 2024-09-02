import os
from utils import FileUtils, StringUtils, JSONUtils
import json
import platform
import datetime

next_type = {
    "project": "dataset",
    "dataset": "counts",
    "counts": "resamples",
    "resamples": "visualizations",
    "visualizations": None
}

prev_type = {
    "project": None,
    "dataset": "project",
    "counts": "dataset",
    "resamples": "counts",
    "visualizations": "resamples"
}

CONSTANTS = {
    "NaN": -1,
    "inf": -2,
}

RESERVED_NAMES = {"data", "All Seq", "All Seq All An", "Sequence Counts"}


class Settings:
    """
    Formerly the primary settings handler for the algorithm.
    Repurposed into a class that handles globals and common names across the algorithm.
    """
    def __init__(self):
        pass

    def getAppDataFolder():
        if platform.system() == "Windows":
            return os.path.join(os.getenv('APPDATA'), "PythonCBAS")
        elif platform.system() == "Darwin":
            return os.path.join(os.getenv('HOME'), "Library", "Application Support", "PythonCBAS")
    
    def getArchiveFolder():
        path = os.path.join(Settings.getAppDataFolder(), "Archives")
        if not os.path.exists(path):
            os.makedirs(path)
        return path
        
    def getDocumentsFolder():
        try:
            if platform.system() == "Windows":
                return os.path.join(os.getenv('USERPROFILE'), "Documents")
            elif platform.system() == "Darwin":
                return os.path.join(os.getenv('HOME'), "Documents")
        except:
            return os.path.expanduser("~")
    
    def getAnimalInfoFormat(self):
        return self.animal_info_format
    
    def getCriterion(self):
        return self.criterion
    
    def getConstants(self):
        return self.constants

    def getCatalogue(self):
        return self.catalogue 
    
    def getAnInfoCol(self, col_name):
        return self.animal_info_format[col_name]
    
    def buildLanguageDict(num_choices, has_modifier, num_contingencies):
        return {
            'NUM_CHOICES': num_choices,
            'HAS_MODIFIER': has_modifier,
            'NUM_CONTINGENCIES': num_contingencies,
        }
    
    def buildCriterionDict(order: int, number, include_failed: bool, allow_redemption: bool):
        return {
            'ORDER': order,
            'NUMBER': number,
            'INCLUDE_FAILED': include_failed,
            'ALLOW_REDEMPTION': allow_redemption
        }
    
    def buildCountsLanguageDict(max_sequence_length: int, straddle_sessions: bool):
        return {
            'MAX_SEQUENCE_LENGTH': max_sequence_length,
            'STRADDLE_SESSIONS': straddle_sessions
        }
    
    def buildDataColumnsDict(session_no_col: int, choice_col: int, contingency_col: int, modifier_col: int):
        return {
            'SESSION_NO_COL': session_no_col,
            'CHOICE_COL': choice_col,
            'CONTINGENCY_COL': contingency_col,
            'MODIFIER_COL': modifier_col
        }
    
    def getColumnOrder():
        return ["Session", "Choice", "Contingency", "Modifier"]
    
    def getCountsFolderPaths(counts_folder):
        return {
            'ALLSEQDIR': os.path.join(counts_folder, 'All Seq'),
            'ALLSEQALLANDIR': os.path.join(counts_folder, 'All Seq All An'),
            'SEQCNTSDIR': os.path.join(counts_folder, 'Sequence Counts'),
        }
    
    def getJSONReference():
        return os.path.join("../json/expected_json_format.json")



class Project:
    def __init__(self):
        self.project_attr = {}
        self.datasets = []

    
        
    def readProject(self, filepath):
        # Fix the JSON File if necessary
        JSONUtils.fixJSONProject(filepath, Settings.getJSONReference(), Settings.getArchiveFolder())
        with open(filepath, 'r') as f:
            data = FileUtils.readJSON(f)

        try:
            self.project_attr = data["project_attr"]
            datasets = data["datasets"]
        except KeyError:
            raise KeyError("The JSON file is not in the correct format.")
        
        for dataset in datasets:
            dataset_obj = DataSet()
            dataset_obj.readDataset(dataset)
            dataset_obj.setParent(self)
            self.datasets.append(dataset_obj)

    def exportProject(self):
        """Generate the dictionary for a project object."""
        project = {
            "project_attr": self.getProjectAttr(),
            "datasets": [dataset.exportDataset() for dataset in self.datasets]
        }
        return project
    
    def writeProject(self):
        filename = self.getName() + ".json" #TODO: Change this to .cbasproj extension
        filepath = os.path.join(self.getDir(), filename)
        self.setDateModified(str(datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")))
        with open(filepath, 'w') as f:
            json.dump(self.exportProject(), f)

    def createProject(self, name, description, datecreated, dir, version):
        # Create a folder for the project
        project_dir = os.path.join(dir, name)
        if not os.path.exists(project_dir):
            os.makedirs(project_dir)
        self.project_attr = {
            "type": "project",
            "name": name,
            "description": description,
            "datecreated": datecreated,
            "datemodified": datecreated,
            "dir": project_dir,
            "version": version
        }

    def addDataset(self, dataset_obj):
        dataset_obj.setParent(self)
        self.datasets.append(dataset_obj)

    def retracePath(self):
        return [self]
    
    def deleteChild(self, child):
        self.datasets.remove(child)
        FileUtils.deleteFolder(child.getDir())

    
    ### GETTER FUNCTIONS ###
    def getType(self) -> str: return "project"
    def getProjectAttr(self) -> dict: return self.project_attr
    def getFilepath(self) -> str: return os.path.join(self.getDir(), self.getName() + ".json")
    def getName(self) -> str: return self.project_attr["name"]
    def getDescription(self) -> str: return self.project_attr["description"]
    def getDir(self) -> str: return self.project_attr["dir"]
    def getProjectVersion(self) -> str: return self.project_attr["version"]
    def getProjectDateCreated(self) -> str: return self.project_attr["datecreated"]
    def getProjectDateModified(self) -> str: return self.project_attr["datemodified"]
    def getChildren(self) -> list: return self.datasets
    def getParent(self) -> None: return None

    ### SETTER FUNCTIONS ###
    def setName(self, name): self.project_attr["name"] = name
    def setDateModified(self, date): self.project_attr["datemodified"] = date
            

class DataSet:
    def __init__(self):
        self.parent = None
        self.counts = []
        self.dataset_settings = None

    def readDataset(self, dataset: dict):
        try:
            assert dataset["type"] == "dataset"
            counts = dataset["counts"]
        except KeyError:
            raise KeyError("JSON error reading in the datasets.")
        
        for count in counts:
            counts_obj = Counts()
            counts_obj.readCounts(count)
            counts_obj.setParent(self)
            self.counts.append(counts_obj)
        
        self.dataset_settings = dataset["dataset_settings"]

    def exportDataset(self):
        """Generate the dictionary for a dataset object."""
        dataset = {
            "type": "dataset",
            "dataset_settings": self.getSettings(),
            "counts": [count.exportCounts() for count in self.counts]
        }
        return dataset
    
    def createDataset(self, name, description, dir, 
                      aninfoname, anInfoColumnNames, anDataColumns, 
                      correlational_possible, num_choices, hasModifier, num_contingencies, num_animals,
                      cohort_file, all_animals_file, all_paths_file):
        self.dataset_settings = {
            "name": name,
            "description": description,
            "dir": dir,
            "aninfoname": aninfoname,
            "anInfoColumnNames": anInfoColumnNames,
            "anDataColumns": anDataColumns,
            "correlational_possible": correlational_possible,
            "language": Settings.buildLanguageDict(num_choices, hasModifier, num_contingencies),
            "num_animals": num_animals,
            "cohort_file": cohort_file,
            "all_animals_file": all_animals_file,
            "all_paths_file": all_paths_file
        }
    
    def addCounts(self, counts_obj):
        counts_obj.setParent(self)
        self.counts.append(counts_obj)

    def writeProject(self):
        self.getParent().writeProject()

    def retracePath(self):
        return self.getParent().retracePath() + [self]
    
    def deleteChild(self, child):
        self.counts.remove(child)
        FileUtils.deleteFolder(child.getDir())

    ### GETTER FUNCTIONS ###
    def getType(self) -> str: return "dataset"
    def getCardInfo(self) -> tuple: return (self.getName(), StringUtils.lastNChars(self.getDir(), 40))
    def getSettings(self) -> dict: return self.dataset_settings
    def getName(self) -> str: return self.dataset_settings["name"]
    def getDescription(self) -> str: return self.dataset_settings["description"]
    def getDir(self) -> str: return self.dataset_settings["dir"]
    def getAnInfoName(self) -> str: return self.dataset_settings["aninfoname"]
    def getAnInfoColumnNames(self) -> list: return self.dataset_settings["anInfoColumnNames"]
    def getAnDataColumns(self) -> dict: return self.dataset_settings["anDataColumns"]
    def correlationalPossible(self) -> bool: return self.dataset_settings["correlational_possible"]
    def getNumAnimals(self) -> int: return self.dataset_settings["num_animals"]
    def getLanguage(self) -> dict: return self.dataset_settings["language"]
    def getNumChoices(self) -> int: return self.getLanguage()["NUM_CHOICES"]
    def getHasModifier(self) -> int: return self.getLanguage()["HAS_MODIFIER"]
    def getNumContingencies(self) -> int: return self.getLanguage()["NUM_CONTINGENCIES"]
    def getCohortFile(self) -> str: return self.dataset_settings["cohort_file"]
    def getAllAnimalsFile(self) -> str: return self.dataset_settings["all_animals_file"]
    def getAllPathsFile(self) -> str: return self.dataset_settings["all_paths_file"]
    def getChildren(self) -> list: return self.counts
    def getParent(self) -> Project: return self.parent

    ### SETTER FUNCTIONS ###
    def setParent(self, parent): self.parent = parent
    def setName(self, name): self.dataset_settings["name"] = name
    def renameDir(self, name): pass # TODO: Implement deep renaming

    
        


class Counts:
    def __init__(self, ):
        self.resamples = []
        self.counts_settings = None
        self.counts_metadata = None
    
    def readCounts(self, counts):
        try:
            assert counts["type"] == "counts"
            resamples = counts["resamples"]
            self.counts_settings = counts["counts_settings"]
            self.counts_metadata = counts["counts_metadata"]
        except KeyError:
            raise KeyError("JSON error reading in the counts.")
        
        for resample in resamples:
            resample_obj = Resamples()
            resample_obj.readResamples(resample)
            resample_obj.setParent(self)
            self.resamples.append(resample_obj)

        

    def exportCounts(self):
        """Generate the dictionary for a counts object."""
        counts = {
            "type": "counts",
            "counts_settings": self.getCountsSettings(),
            "counts_metadata": self.getCountsMetadata(),
            "resamples": [resample.exportResamples() for resample in self.resamples]
        }
        return counts

    def createCounts(self, name: str, dir, description: str, criterion_dict: dict, counts_language_dict: dict, time_taken: float):
        self.counts_settings = {
            "name": name,
            "dir": dir,
            "description": description,
            "criterion": criterion_dict,
            "counts_language": counts_language_dict,
        }
        self.counts_metadata = {
            "time_taken": time_taken,
            "date_created": str(datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
        }
    
    def addResamples(self, resample_obj):
        resample_obj.setParent(self)
        self.resamples.append(resample_obj)

    def retracePath(self):
        return self.getParent().retracePath() + [self]
    
    def deleteChild(self, child):
        self.resamples.remove(child)
        FileUtils.deleteFolder(child.getDir())

    def writeProject(self):
        self.getParent().writeProject()


    ### GETTER FUNCTIONS ###
    def getType(self) -> str: return "counts"
    def getCardInfo(self): return (self.getName(), self.getDescription())
    def getCountsSettings(self) -> dict: return self.counts_settings
    def getName(self) -> str: return self.counts_settings["name"]
    def getDir(self) -> str: return self.counts_settings["dir"]
    def getDescription(self) -> str: return self.counts_settings["description"]
    def getCriterion(self) -> dict: return self.counts_settings["criterion"]
    def getOrder(self) -> list: return self.getCriterion()["ORDER"]
    def getNumber(self) -> int: return self.getCriterion()["NUMBER"]
    def includeFailed(self) -> bool: return self.getCriterion()["INCLUDE_FAILED"]
    def allowRedemption(self) -> bool: return self.getCriterion()["ALLOW_REDEMPTION"]
    def getCountsLanguage(self) -> dict: return self.counts_settings["counts_language"]
    def getMaxSequenceLength(self) -> int: return self.getCountsLanguage()["MAX_SEQUENCE_LENGTH"]
    def straddleSessions(self) -> bool: return self.getCountsLanguage()["STRADDLE_SESSIONS"]
    def getCountsMetadata(self) -> dict: return self.counts_metadata
    def getCountsTimeTaken(self) -> float: return self.counts_metadata["time_taken"]
    def getCountsDateCreated(self) -> str: return self.counts_metadata["date_created"]
    def getChildren(self) -> list: return self.resamples
    def getParent(self) -> DataSet: return self.parent

    ### SETTER FUNCTIONS ###
    def setParent(self, parent:DataSet): self.parent = parent
    def setName(self, name): self.counts_settings["name"] = name



class Resamples:
    def __init__(self, ):
        self.pvalues = []
        self.resample_settings = None
        self.pvalue_settings = None  # [PVALUE]
        self.resamples_metadata = None
        self.pvalues_metadata = None
    
    def readResamples(self, resamples):
        try:
            assert resamples["type"] == "resamples"
            self.resample_settings = resamples["resample_settings"]
            self.pvalue_settings = resamples["pvalue_settings"]  # [PVALUE]
            self.resamples_metadata = resamples["resamples_metadata"]
            self.pvalues_metadata = resamples["pvalues_metadata"]
        except KeyError:
            raise KeyError("JSON error reading in the resamples.")

    def exportResamples(self):
        """Generate the dictionary for a resamples object."""
        resamples = {
            "type": "resamples",
            "resample_settings": self.getResampleSettings(),
            "pvalue_settings": self.getPValueSettings(),
            "resamples_metadata": self.getResamplesMetadata(),
            "pvalues_metadata": self.getPValuesMetadata(),
        }
        return resamples
    
    def createResamples(self, name: str, description: str, directory, correlational: bool, seed, num_resamples: int, contingencies: list, groups: dict,
                       fdp: bool, alpha: float, gamma: float,
                       resamples_time_taken: float, pvalues_time_taken: float):
        reformatted_groups = [[int(num) for num in group] for group in groups] if not correlational else None
        self.resample_settings = {
            "name": name,
            "description": description,
            "dir": directory,
            "correlational": correlational,
            "seed": seed,
            "numresamples": num_resamples,
            "contingencies": contingencies,
            "groups": reformatted_groups
        }
        self.pvalue_settings = {
            "fdp": fdp,
            "alpha": alpha,
            "gamma": gamma
        }
        self.resamples_metadata = {
            "time_taken": resamples_time_taken,
            "date_created": str(datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
        }
        self.pvalues_metadata = {
            "time_taken": pvalues_time_taken,
            "date_created": str(datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
        }
    
    # def addPvaluesObj(self, pvalue_obj):
    #     pvalue_obj.setParent(self)
    #     self.pvalues.append(pvalue_obj)

    def writeProject(self):
        self.getParent().writeProject()

    def retracePath(self):
        return self.getParent().retracePath() + [self]

    ### GETTER FUNCTIONS ###
    def getType(self) -> str: return "resamples"
    def getCardInfo(self): return (self.getName(), self.getDescription())
    def getResampleSettings(self) -> dict: return self.resample_settings
    def getName(self) -> str: return self.resample_settings["name"]
    def getDescription(self) -> str: return self.resample_settings["description"]
    def getDir(self) -> str: return self.resample_settings["dir"]
    def isCorrelational(self) -> bool: return self.resample_settings["correlational"]
    def getSeed(self) -> int: return self.resample_settings["seed"]
    def getNumResamples(self) -> int: return self.resample_settings["numresamples"]
    def getContingencies(self) -> list: return self.resample_settings["contingencies"]
    def getGroups(self) -> dict: return self.resample_settings["groups"]
    def getResamplesMetadata(self) -> dict: return self.resamples_metadata
    def getResamplesTimeTaken(self) -> float: return self.resamples_metadata["time_taken"]
    def getResamplesDateCreated(self) -> str: return self.resamples_metadata["date_created"]
    # P-Value Settings are here for now, since we're combining until we can figure out how to separate them
    def getPValueSettings(self) -> dict: return self.pvalue_settings
    def useFDP(self) -> bool: return self.pvalue_settings["fdp"]
    def getAlpha(self) -> float: return self.pvalue_settings["alpha"]
    def getGamma(self) -> float: return self.pvalue_settings["gamma"]
    def getPValuesMetadata(self) -> dict: return self.pvalues_metadata
    def getPValuesTimeTaken(self) -> float: return self.pvalues_metadata["time_taken"]
    def getPValuesDateCreated(self) -> str: return self.pvalues_metadata["date_created"]
    # End of P-Value Settings
    def getChildren(self) -> list: return self.pvalues
    def getParent(self) -> Counts: return self.parent

    ### SETTER FUNCTIONS ###
    def setParent(self, parent:Counts): self.parent = parent
    def setName(self, name): self.resample_settings["name"] = name


class Visualizations:
    def __init__(self, ):
        pass
    
    def readVisualizations(self, visualizations):
        try:
            assert visualizations["type"] == "visualizations"
            self.visualization_settings = visualizations["visualization_settings"]
        except KeyError:
            raise KeyError("JSON error reading in the visualizations.")
        
    def exportVisualizations(self):
        """Generate the dictionary for a visualizations object."""
        visualizations = {
            "type": "visualizations",
            "pvalueid": self.getVisualizationSettings(),
        }
        return visualizations
    
    def createVisualizations(self):
        pass

    def writeProject(self):
        self.getParent().writeProject()

    def retracePath(self):
        return self.getParent().retracePath() + [self]
    
    ### GETTER FUNCTIONS ###
    def getType(self) -> str: return "visualizations"
    def getCardInfo(self): return None
    def getVisualizationSettings(self) -> dict: return self.visualization_settings
    def getChildren(self) -> list: return []
    def getParent(self) -> Resamples: return self.parent

    ### SETTER FUNCTIONS ###
    def setParent(self, parent:Resamples): self.parent = parent
    def setName(self, name): self.visualization_settings["name"] = name

    


class Preferences:
    def __init__(self, preferences_filepath):
        self.preferences_filepath = preferences_filepath
        self.recentlyOpened = set()

        if not os.path.exists(preferences_filepath):
            self.writePreferences()
        else:
            try:
                self.readPreferences(preferences_filepath)
            except:
                self.resetPreferences()
                self.writePreferences()
    
    def readPreferences(self, filepath):
        with open(filepath, 'r') as f:
            data = FileUtils.readJSON(f)

        try:
            self.recentlyOpened = set(data["recentlyOpened"])
        except KeyError:
            raise KeyError("Problem importing preferences.")
        
    def genPreferenceDict(self):
        """Generate the dictionary for a preferences object."""
        preferences = {
            "recentlyOpened": list(self.recentlyOpened)
        }
        return preferences
    
    def writePreferences(self):
        with open(self.preferences_filepath, 'w') as f:
            json.dump(self.genPreferenceDict(), f)

    def addRecentlyOpened(self, filepath):
        self.recentlyOpened.add(filepath)
        self.writePreferences()

    def removeRecentlyOpened(self, filepath):
        self.recentlyOpened.remove(filepath)
        self.writePreferences()

    def getRecentlyOpened(self):
        return self.recentlyOpened
    
    def exportPreferences(self, filepath):
        with open(filepath, 'w') as f:
            json.dump(self.genPreferenceDict(), f)

    def importPreferences(self, filepath):
        self.readPreferences(filepath)
        self.writePreferences()

    def resetPreferences(self):
        self.recentlyOpened = set()
        self.writePreferences()



        



if __name__ == "__main__":
    project = Project()
    project.createProject("Project1", "hi", "2021-08-25", "Project1", "1.0")
    print(project.exportProject())

    dataset = DataSet()
    dataset.createDataset("Dataset1", "Dataset1", "anInfo.txt", [], [], True, {})
    print(dataset.exportDataset())
    project.addDataset(dataset.getDatasetID(), dataset)

    dataset = DataSet()
    dataset.createDataset("Dataset2", "Dataset2", "anInfo.txt", [], [], True, {})
    print(dataset.exportDataset())
    project.addDataset(dataset.getDatasetID(), dataset)

    project.writeProject()
