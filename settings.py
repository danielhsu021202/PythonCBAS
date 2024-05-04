import os
from utils import FileUtils
import uuid
import json

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
            'OUTPUT': 'output_hex/',
            'ALLSEQDIR': os.path.join('output_hex', 'All Seq'),
            'ALLSEQALLANDIR': os.path.join('output_hex', 'All Seq All An'),
            'SEQCNTSDIR': os.path.join('output_hex', 'Sequence Counts'),
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


    
    def setLanguage(num_choices, modifier_ranges):
        """Sets the language for the experiment. The language is defined by the number of choices and the number of modifiers."""
        NUM_CHOICES = num_choices
        NUM_MODIFIERS = len(modifier_ranges)
        MODIFIER_RANGES = modifier_ranges

    def setCriterion(self, attr_dict: dict):
        """Sets the criterion for the experiment. The criterion is defined by the order, number, whether to include failed trials, and whether to allow redemption."""
        assert attr_dict.keys() == self.criterion.keys()
        self.criterion = attr_dict

    

            

    # def setCriterion(self, order: int, number: int, include_failed: bool, allow_redemption: bool):
    #     """Sets the criterion for the experiment. The criterion is defined by the order, number, whether to include failed trials, and whether to allow redemption."""
    #     self.criterion['ORDER'] = order
    #     self.criterion['NUMBER'] = number
    #     self.criterion['INCLUDE_FAILED'] = include_failed
    #     self.criterion['ALLOW_REDEMPTION'] = allow_redemption



class Project:
    def __init__(self):
        self.projectid = None
        self.project_attr = {}
        self.datasets = {}
        
    def readProject(self, filepath):
        with open(filepath, 'r') as f:
            data = FileUtils.readJSON(f)

        try:
            self.project_attr = data["project_attr"]
            datasets = data["datasets"]
        except KeyError:
            raise KeyError("The JSON file is not in the correct format.")
        
        self.projectid = self.project_attr["projectid"]
        
        for dataset in datasets:
            dataset_obj = DataSet()
            dataset_obj.readDataset(dataset)
            datasetid = dataset_obj.getDatasetID()
            self.datasets[datasetid] = dataset_obj

    def exportProject(self):
        """Generate the dictionary for a project object."""
        project = {
            "project_attr": self.getProjectAttr(),
            "datasets": [dataset.exportDataset() for dataset in self.datasets.values()]
        }
        return project
    
    def writeProject(self):
        filename = self.getProjectName() + ".json" #TODO: Change this to .cbasproj extension
        filepath = os.path.join(self.getProjectDir(), filename)
        with open(filepath, 'w') as f:
            json.dump(self.exportProject(), f)

    def createProject(self, name, description, datecreated, dir, version):
        self.projectid = str(uuid.uuid4())
        self.project_attr = {
            "type": "project",
            "projectid": self.projectid,
            "name": name,
            "description": description,
            "datecreated": datecreated,
            "datemodified": datecreated,
            "dir": dir,
            "version": version
        }

    def addDataset(self, datasetid, dataset_obj):
        dataset_obj.setProjectID(self.projectid)
        self.datasets[datasetid] = dataset_obj

    
    ### GETTER FUNCTIONS ###
    def getProjectAttr(self) -> dict: return self.project_attr
    def getProjectID(self) -> str: return self.projectid
    def getProjectName(self) -> str: return self.project_attr["name"]
    def getProjectDir(self) -> str: return self.project_attr["dir"]
    def getProjectVersion(self) -> str: return self.project_attr["version"]
    def getProjectDateCreated(self) -> str: return self.project_attr["datecreated"]
    def getProjectDateModified(self) -> str: return self.project_attr["datemodified"]
            

class DataSet:
    def __init__(self):
        self.datasetid = None
        self.projectid = None
        self.counts = {}
        self.dataset_settings = None

    def readDataset(self, dataset: dict):
        try:
            assert dataset["type"] == "dataset"
            self.datasetid = dataset["datasetid"]
            self.projectid = dataset["projectid"]
            counts = dataset["counts"]
        except KeyError:
            raise KeyError("Error reading in the datasets.")
        
        for count in counts:
            counts_obj = Counts()
            counts_obj.readCounts(count)
            countsid = counts_obj.getCountsID()
            self.counts[countsid] = counts_obj
        
        self.dataset_settings = dataset["dataset_settings"]

    def exportDataset(self):
        """Generate the dictionary for a dataset object."""
        dataset = {
            "type": "dataset",
            "datasetid": self.getDatasetID(),
            "projectid": self.getProjectID(),
            "dataset_settings": self.getSettings(),
            "counts": [count.exportCounts() for count in self.counts.values()]
        }
        return dataset
    
    def createDataset(self, name, dir, aninfoname, anInfoColumnNames, anDataColumnNames, correlational_possible, language):
        self.datasetid = str(uuid.uuid4())
        self.dataset_settings = {
            "name": name,
            "dir": dir,
            "aninfoname": aninfoname,
            "anInfoColumnNames": anInfoColumnNames,
            "anDataColumnNames": anDataColumnNames,
            "correlational_possible": correlational_possible,
            "language": language
        }
        

    def getProjectID(self):
        return self.projectid

    def getDatasetID(self):
        return self.datasetid
    
    def addCountsObj(self, countsid, counts_obj):
        self.counts[countsid] = counts_obj

    ### GETTER FUNCTIONS ###
    def getSettings(self) -> dict: return self.dataset_settings
    def getName(self) -> str: return self.dataset_settings["name"]
    def getDir(self) -> str: return self.dataset_settings["dir"]
    def getAnInfoName(self) -> str: return self.dataset_settings["aninfoname"]
    def getAnInfoColumnNames(self) -> list: return self.dataset_settings["anInfoColumnNames"]
    def getAnDataColumnNames(self) -> list: return self.dataset_settings["anDataColumnNames"]
    def correlationalPossible(self) -> bool: return self.dataset_settings["correlational_possible"]
    def getLanguage(self) -> dict: return self.dataset_settings["language"]
    def getNumChoices(self) -> int: return self.getLanguage()["num_choices"]
    def getNumModifiers(self) -> int: return self.getLanguage()["num_modifiers"]
    def getNumContingencies(self) -> int: return self.getLanguage()["num_contingencies"]

    ### SETTER FUNCTIONS ###
    def setProjectID(self, projectid): self.projectid = projectid
    
        


class Counts:
    def __init__(self, ):
        self.countsid = None
        self.datasetid = None
        self.resamples = {}
        self.counts_settings = None
    
    def readCounts(self, counts):
        try:
            self.countsid = counts["countsid"]
            self.datasetid = counts["datasetid"]
            resamples = counts["resamples"]
        except KeyError:
            raise KeyError("Error reading in the counts.")
        
        for resample in resamples:
            resample_obj = Resamples()
            resample_obj.readResamples(resample)
            resampleid = resample_obj.getResampleID()
            self.resamples[resampleid] = resample_obj

        self.counts_settings = counts["counts_settings"]

    def exportCounts(self):
        """Generate the dictionary for a counts object."""
        counts = {
            "type": "counts",
            "countsid": self.getCountsID(),
            "datasetid": self.getDatasetID(),
            "counts_settings": self.getCountsSettings(),
            "criterion": self.getCriterion(),
            "max_seq_len": self.getMaxSequenceLength(),
            "outputdir": self.getOutputDir(),
            "straddle_sessions": self.straddleSessions(),
            "resamples": [resample.exportResamples() for resample in self.resamples.values()]
        }
        return counts

    def getCountsID(self): return self.countsid
    
    def getDatasetID(self): return self.datasetid
    
    def addResamplesObj(self, resampleid, resample_obj):
        self.resamples[resampleid] = resample_obj

    ### GETTER FUNCTIONS ###
    def getCountsSettings(self) -> dict: return self.counts_settings
    def getCriterion(self) -> dict: return self.counts_settings["criterion"]
    def getOrder(self) -> list: return self.getCriterion()["order"]
    def getNumber(self) -> int: return self.getCriterion()["number"]
    def includeFailed(self) -> bool: return self.getCriterion()["include_failed"]
    def allowRedemption(self) -> bool: return self.getCriterion()["allow_redemption"]
    def getMaxSequenceLength(self) -> int: return self.counts_settings["max_seq_len"]
    def getOutputDir(self) -> str: return self.counts_settings["outputdir"]
    def straddleSessions(self) -> bool: return self.counts_settings["straddle_sessions"]



class Resamples:
    def __init__(self, ):
        self.resampleid = None
        self.countsid = None
        self.pvalues = {}
        self.resample_settings = None
    
    def readResamples(self, resamples):
        try:
            self.resampleid = resamples["resampleid"]
            self.countsid = resamples["countsid"]
            pvalues = resamples["pvalues"]
        except KeyError:
            raise KeyError("Error reading in the resamples.")
        
        for pvalue in pvalues:
            pvalue_obj = Pvalues()
            pvalue_obj.readPvalues(pvalue)
            pvalueid = pvalue_obj.getPvalueID()
            self.pvalues[pvalueid] = pvalue_obj
        
        self.resample_settings = resamples["resample_settings"]

    def exportResamples(self):
        """Generate the dictionary for a resamples object."""
        resamples = {
            "type": "resample",
            "resampleid": self.getResampleID(),
            "countsid": self.getCountsID(),
            "resample_settings": self.getResampleSettings(),
            "pvalues": [pvalue.exportPvalues() for pvalue in self.pvalues.values()]
        }
        return resamples
    
    def getResampleID(self): return self.resampleid

    def getCountsID(self): return self.countsid
    
    def addPvaluesObj(self, pvalueid, pvalue_obj):
        self.pvalues[pvalueid] = pvalue_obj

    ### GETTER FUNCTIONS ###
    def getResampleSettings(self) -> dict: return self.resample_settings
    def getSeed(self) -> int: return self.resample_settings["seed"]
    def getNumResamples(self) -> int: return self.resample_settings["numresamples"]
    def getContingencies(self) -> list: return self.resample_settings["contingencies"]
    def getGroups(self) -> dict: return self.resample_settings["groups"]
    
    


class Pvalues:
    def __init__(self, ):
        self.pvalueid = None
        self.resampleid = None
        self.visualizations = {}
        self.pvaluesettings = None
    
    def readPvalues(self, pvalues):
        try:
            self.pvalueid = pvalues["pvalueid"]
            resampleid = pvalues["resampleid"]
            visualizations = pvalues["visualizations"]
        except KeyError:
            raise KeyError("Error reading in the pvalues.")
        
        for visualization in visualizations:
            visualization_obj = Visualizations()
            visualization_obj.readVisualizations(visualization)
            visualizationid = visualization_obj.getVisualizationID()
            self.visualizations[visualizationid] = visualization_obj

    def exportPvalues(self):
        """Generate the dictionary for a pvalues object."""
        pvalues = {
            "type": "pvalues",
            "pvalueid": self.getPvalueID(),
            "resampleid": self.getResampleID(),
            "pvaluesettings": self.getPvaluesSettings(),
            "visualizations": [visualization.exportVisualizations() for visualization in self.visualizations.values()]
        }
        return pvalues

    def getPvalueID(self):
        return self.pvalueid
    
    def getResampleID(self):
        return self.resampleid
    
    ### GETTER FUNCTIONS ###
    def getPvaluesSettings(self) -> dict: return self.pvaluesettings
    def useFDP(self) -> bool: return self.pvaluesettings["fdp"]
    def getAlpha(self) -> float: return self.pvaluesettings["alpha"]
    def getGamma(self) -> float: return self.pvaluesettings["gamma"]

    
    def addVisualizationsObj(self, visualizationid, visualization_obj):
        self.visualizations[visualizationid] = visualization_obj


class Visualizations:
    def __init__(self, ):
        self.pvalueid = None
        self.visualizationid = None
    
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

    def getPvalueID(self):
        return self.pvalueid

    def getVisualizationID(self):
        return self.visualizationid
    


class Preferences:
    def __init__(self, ):
        pass
    
    def readPreferences(self, preferences):
        pass



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
