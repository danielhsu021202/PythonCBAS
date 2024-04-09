import numpy as np
import re
import pickle
import gzip
from scipy.sparse import csr_matrix

class HexUtils:

    def numHexDigits(num: int):
        return len(hex(num)) - 2
    
    def getSeqNumHexNum(num: int, cont, length, LANGUAGE):
        """Constructs the hex string for the sequence number given the number of digits needed to encode the number of contingencies and the maximum sequence length.
        Returns the integer represented by the hex string."""
        nex_num = hex(num)[2:]
        # The hex string for contingency and length, padded with zeros to the appropriate number of digits
        hex_cont = hex(cont)[2:].zfill(HexUtils.numHexDigits(LANGUAGE['NUM_CONTINGENCIES']))
        hex_length = hex(length)[2:].zfill(HexUtils.numHexDigits(LANGUAGE['MAX_SEQUENCE_LENGTH']))
        return int(hex_cont + hex_length + nex_num, 16)
    

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
    
    

class FileUtils:

    def getMatrix(file, delimiter=',', dtype=int):
        """Takes a text file and returns a numpy matrix"""
        # Force it to be 2D even if there's only one row
        return np.atleast_2d(np.genfromtxt(file, delimiter=delimiter, dtype=dtype))
    
    def writeMatrix(file, mat, filetype='txt', delimiter=',', fmt='%d'):
        """Writes a numpy matrix to a text file. Don't write an extra line"""
        np.savetxt(file, mat, delimiter=delimiter, fmt=fmt)

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
            
class MatrixUtils:

    def isSparse(matrix, threshold=0.2):
        """Returns True if the matrix is sparse, False otherwise."""
        sparsity = np.count_nonzero(matrix) / matrix.size
        return sparsity < threshold
    
    def csrCompress(matrix):
        """Compresses a matrix to a CSR matrix."""
        return csr_matrix(matrix)
    
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
