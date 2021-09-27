import pandas as pd
import numpy as np

class EcommerceDataset:
    def __init__(self, File):
        self.data = pd.read_csv(File)
    
    def returnFile(self):
        return np.array(self.data.iloc[:, 3:7], dtype=np.float32),\
            np.array(self.data.iloc[:, -1], dtype=np.float32)

    def __len__(self):
        return len(self.data), len(self.data[0])