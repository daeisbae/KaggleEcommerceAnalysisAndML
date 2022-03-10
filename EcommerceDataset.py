import pandas as pd
from sklearn.model_selection import train_test_split

class EcommerceDataset:
    def __init__(self, FileLocation, test_size = 0.2):
        self.data = pd.read_csv(FileLocation)
        self.state = 'train'
        self.test_size = test_size
        self.__DataSplit()

    def __DataSplit(self):
        """
        Split the data into train set, test set
        default test size = 0.2
        """
        self.train_x, self.test_x, self.train_y, self.test_y =\
             train_test_split(self.data.iloc[:, [3, 4, 6]], self.data.iloc[:, -1], test_size=self.test_size)

    def ChangeState(self):
        """
        Change which dataset to return when return_data is called
        """
        if self.state == 'train':
            self.state = 'test'
        else:
            self.state = 'train'

    def return_data(self):
        """
        Return dataset based on current self.state
        """
        if self.state == 'train':
            return self.train_x, self.train_y
        return self.test_x, self.test_y

    def return_all_data(self):
        """
        Return all x, y dataset
        """
        return self.data.iloc[:, [3, 4, 6]], self.data.iloc[:, -1]

    def __len__(self):
        return len(self.data), len(self.data[0])