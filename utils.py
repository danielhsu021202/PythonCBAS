import numpy as np
import re
import os
import shutil
import pickle
import json
import gzip
import datetime
import subprocess
import requests
from copy import deepcopy
from scipy.sparse import csr_matrix

class StringUtils:
    
    def parseComplexRange(range_str: str):
        """Parses a complex range string and returns a tuple of the range."""
        try:
            tokens = range_str.split(",")
            values = []
            for token in tokens:
                if "-" in token:
                    values.extend(range(int(token.split("-")[0]), int(token.split("-")[1]) + 1))
                else:
                    values.append(int(token))
            return values
        except:
            return None
        
    def andSeparateList(lst: list, include_verb=False) -> str:
        """Returns a string with the elements of the list separated by 'and'."""
        if len(lst) == 0:
            return ""
        if len(lst) == 1:
            return lst[0] + (" is" if include_verb else "")
        elif len(lst) == 2:
            return lst[0] + " and " + lst[1] + (" are" if include_verb else "")
        else:
            return ", ".join(lst[:-1]) + ", and " + lst[-1] + (" are" if include_verb else "")

    def andSeparateStr(string: str, include_verb=False) -> str:
        """Returns a string with the elements of the string separated by 'and'."""
        return StringUtils.andSeparateList([s.strip() for s in string.split(",")], include_verb)
    
    def capitalizeFirstLetter(s: str) -> str:
        """Capitalizes the first letter of a string."""
        return s[0].upper() + s[1:]
    
    def lastNChars(s: str, n: int) -> str:
        """Returns the last n characters of a string. Precede with elipsis if n is less than the length of the string."""
        return s if len(s) <= n else "..." + s[-(n - 3):]
    
    

class FileUtils:

    def getMatrix(file, delimiter=',', dtype=int, limit_rows=None) -> np.ndarray:
        """Takes a text file and returns a numpy matrix"""
        return np.atleast_2d(np.genfromtxt(file, delimiter=delimiter, dtype=dtype, max_rows=limit_rows)) # Force it to be 2D even if there's only one row
    
    def writeMatrix(file, mat, filetype='txt', delimiter=',', fmt='%d'):
        """Writes a numpy matrix to a text file. Don't write an extra line"""
        with open(file, 'w') as f:
            np.savetxt(f, mat, delimiter=delimiter, fmt=fmt)

    def getMatrixNpy(file):
        """Loads a numpy matrix from a .npy file."""
        return np.load(file)
    
    def writeMatrixNpy(file, mat):
        """Writes a numpy matrix to a .npy file."""
        np.save(file, mat)

    def isCompressed(filepath):
        """Returns True if the file is compressed, False otherwise"""
        try:
            with open(filepath, 'rb') as f:
                return f.read(2) == b'\x1f\x8b'
        except OSError:
            return False

    def pickleObj(obj, filepath, compress=False):
        """Pickle an object to a file. User can choose to compress the file."""
        if compress:
            with gzip.open(filepath, 'wb') as f:
                pickle.dump(obj, f)
        else:
            with open(filepath, 'wb') as f:
                pickle.dump(obj, f)

    def unpickleObj(filepath):
        """Unpickle an object from a file. Decompress if necessary."""
        if FileUtils.isCompressed(filepath):
            with gzip.open(filepath, 'rb') as f:
                return pickle.load(f)
        else:
            with open(filepath, 'rb') as f:
                return pickle.load(f)
    
    def writeJSON(filepath, obj):
        """Write a JSON object to a file."""
        json.dump(obj, filepath)

    def readJSON(filepath):
        """Read a JSON object from a file."""
        return json.load(filepath)
    
    def validFile(filepath):
        """Returns True if the file is not hidden (starts with '.')"""
        if os.path.isfile(filepath):
            return not os.path.basename(filepath).startswith('.')
        
    def deleteFile(filepath):
        """Deletes a file."""
        try:
            os.remove(filepath)
        except:
            pass
        
    def deleteFolder(folder):
        """Deletes a folder and all its contents."""
        try:
            shutil.rmtree(folder)
        except:
            pass

    def archiveFile(file, archive_folder):
        # Rename the file to append the time of update after an underscore, retaining the extension
        new_name = os.path.basename(file).split(".")[0] + "_" + datetime.datetime.now().strftime("%Y_%m_%d_%H_%M_%S") + "." + os.path.basename(file).split(".")[1]
        # Copy to archive folder
        archive_path = os.path.join(archive_folder, new_name)
        shutil.copy(file, archive_path)
        return archive_path

        



class JSONUtils:
    def getKeyName(key):
        """
        Given a key, returns the key name that should be used for deep checking.
        This is needed because of semantics.
        For datasets, singular is "dataset" and plural is "datasets";
        But for everything else, singular is the same as plural (e.g. sing. "counts" = plur. "counts")
        """
        if key == "datasets": return "dataset"
        else: return key

    def fixJSON(data, ref_dict, overall_ref):
        """
        Helper function for fixJSONProject that recursively fixes a JSON file by adding missing keys from a reference JSON file.
        """
        for key in ref_dict:
            if key not in data:
                data[key] = ref_dict[key]
            elif type(ref_dict[key]) == dict:
                JSONUtils.fixJSON(data[key], ref_dict[key], overall_ref)
            elif type(ref_dict[key]) == list:
                assert key in data and type(data[key]) == list
                for dict_obj in data[key]:
                    key_name = JSONUtils.getKeyName(key)
                    JSONUtils.fixJSON(dict_obj, overall_ref[key_name], overall_ref)
                

    def fixJSONProject(filepath, reference, archive_folder):
        """
        Fixes a JSON file by adding missing keys from a reference JSON file for project files.
        Deep checks keys that have "attr" or "settings" as part of their key.
        """
        archive = FileUtils.archiveFile(filepath, archive_folder)
        with open(filepath, 'r') as f:
            data = json.load(f)
        with open(reference, 'r') as f:
            ref = json.load(f)
        orig_data = deepcopy(data)
        JSONUtils.fixJSON(data, ref["project"], ref)
        modified = data != orig_data  # Check if the data was modified
        if modified:
            # Write the fixed JSON back to the file
            with open(filepath, 'w') as f:
                FileUtils.writeJSON(f, data)
        else:
            # Delete the archive
            FileUtils.deleteFile(archive)
        


    def fixJSONPreferences(filepath, reference):
        """
        Fixes a JSON file by adding missing keys from a reference JSON file for preferences files.
        """
        with open(filepath, 'r') as f:
            data = json.load(f)
        with open(reference, 'r') as f:
            ref = json.load(f)
        for key in ref:
            if key not in data:
                data[key] = ref[key]
                print(f"Added key {key}")
        

    

            
class MatrixUtils:

    def isSparse(matrix, threshold=0.2):
        """Returns True if the matrix is sparse, False otherwise."""
        sparsity = np.count_nonzero(matrix) / matrix.size
        return sparsity < threshold
    
    def csrCompress(matrix, dtype):
        """Compresses a matrix to a CSR matrix."""
        return csr_matrix(matrix, dtype=dtype)
    
    def csrDecompress(csr_matrix: csr_matrix):
        """Decompresses a CSR matrix to a dense matrix."""
        return csr_matrix.toarray()
    
    def getRow(matrix, row):
        """Returns the row of a matrix."""
        return matrix[row]
    
    def getCol(matrix, col):
        """Returns the column of a matrix."""
        return matrix[:, col]



class ListUtils:

    def naturalSort(l, key=lambda x: x):
        """Sorts the given list in natural order."""
        return sorted(l, key=lambda x: [int(c) if c.isdigit() else c for c in re.split('([0-9]+)', key(x))])
    
    def flatten(l):
        """Flattens a list."""
        return [item for sublist in l for item in sublist]
    
    def unique(l):
        """Returns the unique elements of a list."""
        return list(set(l))
    
    def count(l, item):
        """Returns the number of occurrences of an item in a list."""
        return l.count(item)
    
    def countUnique(l):
        """Returns the number of unique elements in a list."""
        return len(ListUtils.unique(l))
    
    def countAll(l):
        """Returns a dictionary of the number of occurrences of each unique element in a list."""
        return {item: l.count(item) for item in ListUtils.unique(l)}
    
    def countAllSorted(l):
        """Returns a dictionary of the number of occurrences of each unique element in a list, sorted by the element."""
        return {k: v for k, v in sorted(ListUtils.countAll(l).items())}
    
class TimeUtils:
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

class WebUtils:
    plotly_js_path = os.path.abspath("js/plotly-2.32.0.min.js")
    plotly_latest_js_url = "https://cdn.plot.ly/plotly-latest.min.js"

    def fetchLatestPlotlyJS():
        if not WebUtils.check_internet_connection():
            raise Exception("No internet connection to fetch plotly.js file.")
        if not os.path.exists("js"):
            os.mkdir("js")
        if not os.path.exists(WebUtils.plotly_js_path):
            # Fetch the plotly.js file
            with open(WebUtils.plotly_js_path, 'wb') as f:
                f.write(requests.get(WebUtils.plotly_latest_js_url).content)


    def check_internet_connection():
        try:
            subprocess.check_output(["ping", "-c", "1", "www.google.com"])
            return True
        except subprocess.CalledProcessError:
            return False
        
    def htmlStartPlotly():
        return f"""
            <html>
            <head>
                <script src="{WebUtils.plotly_latest_js_url}"></script>
            </head>
            <body>
        """
    
    def htmlEnd():
        return """
            </body>
            </html>
        """


        

# Test Fix JSON

if __name__ == "__main__":
    JSONUtils.fixJSONProject("TestProject/Test Project Update JSON/Test Project Update JSON.json", "json/expected_format_project.json")
