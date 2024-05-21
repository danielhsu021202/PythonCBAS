import os
from utils import FileUtils, StringUtils
import uuid
import json
import datetime

next_type = {
    "project": "dataset",
    "dataset": "counts",
    "counts": "resamples",
    # "resamples": "pvalues",
    # "pvalues": "visualizations",
    "resamples": "visualizations",
    "visualizations": None
}

prev_type = {
    "project": None,
    "dataset": "project",
    "counts": "dataset",
    "resamples": "counts",
    # "pvalues": "resamples",
    # "visualizations": "pvalues"
    "visualizations": "resamples"
}

CONSTANTS = {
    "NaN": None,
    "inf": float("inf"),
}


class Settings:
    """
    Formerly the primary settings handler for the algorithm.
    Repurposed into a class that handles globals and common names across the algorithm.
    """
    def __init__(self):
        pass
    
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


    
    def setLanguage(num_choices, modifier_ranges):
        """Sets the language for the experiment. The language is defined by the number of choices and the number of modifiers."""
        NUM_CHOICES = num_choices
        NUM_MODIFIERS = len(modifier_ranges)
        MODIFIER_RANGES = modifier_ranges

    def setCriterion(self, attr_dict: dict):
        """Sets the criterion for the experiment. The criterion is defined by the order, number, whether to include failed trials, and whether to allow redemption."""
        assert attr_dict.keys() == self.criterion.keys()
        self.criterion = attr_dict



class Project:
    def __init__(self):
        self.project_attr = {}
        self.datasets = []
        
    def readProject(self, filepath):
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
            raise KeyError("Error reading in the datasets.")
        
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
    
        


class Counts:
    def __init__(self, ):
        self.resamples = []
        self.counts_settings = None
    
    def readCounts(self, counts):
        try:
            resamples = counts["resamples"]
        except KeyError:
            raise KeyError("Error reading in the counts.")
        
        for resample in resamples:
            resample_obj = Resamples()
            resample_obj.readResamples(resample)
            resample_obj.setParent(self)
            self.resamples.append(resample_obj)

        self.counts_settings = counts["counts_settings"]

    def exportCounts(self):
        """Generate the dictionary for a counts object."""
        counts = {
            "type": "counts",
            "counts_settings": self.getCountsSettings(),
            "resamples": [resample.exportResamples() for resample in self.resamples]
        }
        return counts

    def createCounts(self, name: str, dir, description: str, criterion_dict: dict, counts_language_dict: dict):
        self.counts_settings = {
            "name": name,
            "dir": dir,
            "description": description,
            "criterion": criterion_dict,
            "counts_language": counts_language_dict,
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
    def getChildren(self) -> list: return self.resamples
    def getParent(self) -> DataSet: return self.parent

    ### SETTER FUNCTIONS ###
    def setParent(self, parent:DataSet): self.parent = parent



class Resamples:
    def __init__(self, ):
        self.pvalues = []
        self.resample_settings = None
        self.pvaluesettings = None  # [PVALUE]
    
    def readResamples(self, resamples):
        # try:
        #     pvalues = resamples["pvalues"]
        # except KeyError:
        #     raise KeyError("Error reading in the resamples.")
        
        # for pvalue in pvalues:
        #     pvalue_obj = Pvalues()
        #     pvalue_obj.readPvalues(pvalue)
        #     pvalue_obj.setParent(self)
        #     self.pvalues.append(pvalue_obj)
        
        self.resample_settings = resamples["resample_settings"]
        self.pvaluesettings = resamples["pvalue_settings"]  # [PVALUE]

    def exportResamples(self):
        """Generate the dictionary for a resamples object."""
        resamples = {
            "type": "resamples",
            "resample_settings": self.getResampleSettings(),
            "pvalue_settings": self.getPValueSettings(),
            # "pvalues": [pvalue.exportPvalues() for pvalue in self.pvalues]
        }
        return resamples
    
    def createResamples(self, name: str, description: str, directory, correlational: bool, seed, num_resamples: int, contingencies: list, groups: dict,
                       fdp: bool, alpha: float, gamma: float):
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
        self.pvaluesettings = {
            "fdp": fdp,
            "alpha": alpha,
            "gamma": gamma
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
    # P-Value Settings are here for now, since we're combining until we can figure out how to separate them
    def getPValueSettings(self) -> dict: return self.pvaluesettings
    def useFDP(self) -> bool: return self.pvaluesettings["fdp"]
    def getAlpha(self) -> float: return self.pvaluesettings["alpha"]
    def getGamma(self) -> float: return self.pvaluesettings["gamma"]
    # End of P-Value Settings
    def getChildren(self) -> list: return self.pvalues
    def getParent(self) -> Counts: return self.parent

    ### SETTER FUNCTIONS ###
    def setParent(self, parent:Counts): self.parent = parent

    
    


# class Pvalues:
#     def __init__(self, ):
#         self.visualizations = []
#         self.pvaluesettings = None
    
#     def readPvalues(self, pvalues):
#         try:
#             visualizations = pvalues["visualizations"]
#         except KeyError:
#             raise KeyError("Error reading in the pvalues.")
        
#         for visualization in visualizations:
#             visualization_obj = Visualizations()
#             visualization_obj.readVisualizations(visualization)

#             self.visualization_obj.setParent(self)
#             self.visualizations.append(visualization_obj)

#     def exportPvalues(self):
#         """Generate the dictionary for a pvalues object."""
#         pvalues = {
#             "type": "pvalues",
#             "pvaluesettings": self.getPvaluesSettings(),
#             "visualizations": [visualization.exportVisualizations() for visualization in self.visualizations]
#         }
#         return pvalues
    
#     ### GETTER FUNCTIONS ###
#     def getType(self) -> str: return "pvalues"
#     def getPvaluesSettings(self) -> dict: return self.pvaluesettings
#     def useFDP(self) -> bool: return self.pvaluesettings["fdp"]
#     def getAlpha(self) -> float: return self.pvaluesettings["alpha"]
#     def getGamma(self) -> float: return self.pvaluesettings["gamma"]
#     def getChildren(self) -> list: return self.visualizations
#     def getParent(self) -> Resamples: return self.parent

#     ### SETTER FUNCTIONS ###
#     def setParent(self, parent:Resamples): self.parent = parent

    
#     def addVisualizationsObj(self, visualization_obj):
#         visualization_obj.setParent(self)
#         self.visualizations.append(visualization_obj)


class Visualizations:
    def __init__(self, ):
        pass
    
    def readVisualizations(self, visualizations):
        try:
            self.pvalueid = visualizations["pvalueid"]
            self.visualizationid = visualizations["visualizationid"]
        except KeyError:
            raise KeyError("Error reading in the visualizations.")
        
    def exportVisualizations(self):
        """Generate the dictionary for a visualizations object."""
        visualizations = {
            "type": "visualizations",
            "pvalueid": self.getPvalueID(),
            "visualizationid": self.getVisualizationID()
        }
        return visualizations
    


class Preferences:
    def __init__(self, preferences_filepath):
        self.preferences_filepath = preferences_filepath
        self.recentlyOpened = set()
    
    def readPreferences(self, filepath):
        with open(filepath, 'r') as f:
            data = FileUtils.readJSON(f)

        try:
            self.recentlyOpened = set(data["recentlyOpened"])
        except KeyError:
            raise KeyError("Problem importing preferences.")
        
    def exportPreferences(self):
        """Generate the dictionary for a preferences object."""
        preferences = {
            "recentlyOpened": list(self.recentlyOpened)
        }
        return preferences
    
    def writePreferences(self):
        with open(self.preferences_filepath, 'w') as f:
            json.dump(self.exportPreferences(), f)

    def addRecentlyOpened(self, filepath):
        self.recentlyOpened.add(filepath)
        self.writePreferences()

    def getRecentlyOpened(self):
        return self.recentlyOpened



        



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
