import os
import numpy as np
from utils import FileUtils, MatrixUtils

from PyQt6.QtWidgets import QMessageBox

        


class CBASFile:
    file_types = {
        'MATRIX': 'MATRIX',
        'DATAFRAME': 'DATAFRAME',
        'ALLSEQ': 'ALLSEQ',
        'ALLSEQALLAN': 'ALLSEQALLAN',
        'SEQCNTS': 'SEQCNTS',
        'SEQRATES': 'SEQRATES',
        'RESAMPLED': 'RESAMPLED',
        'SIGSEQS': 'SIGSEQS',
    }

    headers = {
        'MATRIX': None,
        'DATAFRAME': None,
        'ALLSEQ': None,
        'ALLSEQALLAN': ['Subject No.', 'Trial No.', 'Seq No.'],
        'SEQCNTS': None,
        'SEQRATES': None,
        'RESAMPLED': None,
        'SIGSEQS': ['P-Value', 'Sequence', 'Contingency', 'Length', 'Local Seq. No.', 'Positively Correlated'],
        
    }

    compression_formats = {
        'UNCOMPRESSED': 0,
        'GZIP': 1,
        'CSR': 2,
    }

    def __init__(self, name, data, info=None, type=None, col_headers: list=None):
        self.name = name
        self.info = info
        self.type = type
        self.data = data
        self.col_headers = col_headers
        self.compression = 0  # This will be set when the file is saved

    def getType(self):
        return self.type

    def getData(self):
        return self.data
    
    def getInfo(self):
        return self.info
    
    def getColumnHeaders(self):
        if self.col_headers is None:
            if self.type is None or CBASFile.headers[self.type] is None:
                return [f"c{i}" for i in range(self.data.shape[1])]
            return CBASFile.headers[self.type]
        return self.col_headers
    


    def saveFile(self, location, use_sparsity_csr=False, use_gzip=False, dtype=int):
        """Pickle this object to a file"""
        # Ensure valid directory
        if not os.path.exists(location):
            os.makedirs(location)
        # Construct the filepath by joining the location and the name, with the extension .cbas
        filepath = os.path.join(location, self.name + '.cbas')

        if use_sparsity_csr:
            assert type(self.data) == np.ndarray
            if MatrixUtils.isSparse(self.data):
                self.data = MatrixUtils.csrCompress(self.data, dtype)
                self.compression = CBASFile.compression_formats['CSR']
            else:
                self.compression = CBASFile.compression_formats['UNCOMPRESSED']
        else:
            self.compression = CBASFile.compression_formats['UNCOMPRESSED']


        FileUtils.pickleObj(self, filepath, compress=use_gzip)

    def loadFile(filepath):
        """Unpickle a file, decompress if necessary, and return the object"""
        file = FileUtils.unpickleObj(filepath)
        assert type(file) == CBASFile
        if file.compression == CBASFile.compression_formats['CSR']:
            file.data = MatrixUtils.csrDecompress(file.data)
        return file

    def export(self, filepath, type="csv"):
        """Exports the file to the given type"""
        try:
            if type == "csv":
                np.savetxt(filepath, self.data, delimiter=",", fmt='%s')
            elif type == "txt":
                np.savetxt(filepath, self.data, fmt='%s')
            else:
                raise ValueError(f"Export type {type} not supported.")
        except Exception as e:
            QMessageBox.warning(None, "Export Error", f"An error occurred while exporting the file: {e}")
            return


    def __eq__(self, other):
        pass
        




