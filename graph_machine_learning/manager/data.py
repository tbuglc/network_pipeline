import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder

# Define a DataManager class to handle loading and cleaning of the training and testing data
class DataManager:
    def __init__(self, min_sample_label: int) -> None:
        """
        Initializes a new instance of DataManager class.

        Args:
        - min_sample_label (int): minimum number of samples for each class label.

        Returns:
        - None.
        """
        # Load the training and testing data from CSV files into Pandas dataframes
        self.training_set = pd.read_csv('data/data.csv', index_col=False)
        self.min_sample_label = min_sample_label

    def get_data(self) -> tuple:
        """
        Gets and cleans the training data, and splits it into training and validation sets.

        Args:
        - None.

        Returns:
        - tuple: a tuple containing the training and validation data as numpy arrays,
        the list of class labels, and the minimum number of samples for each label.
        """
        self.training_set.fillna(0, inplace=True)

        data = self.training_set.iloc[:,1:]
        # print(data.head())
        # Extract the input features (pixels) and target variable (species) from the training data
        x = np.array(self.training_set.iloc[:, 3:-3].values)

        target = self.training_set.iloc[:,-3:].values
       
        t = target
        # print(t)
        # species = list(set(target))

        # Convert the target variable to integer labels (indices of species list)
        # t = []
        # for i, tg in enumerate(target):
        #     t.append(species.index(tg))
        # t = np.array(t)

        # Split the training data into training and validation sets
        x_train, x_val, t_train, t_val = train_test_split(
            x, t, test_size=0.2, random_state=42)

        # Clean the training data by removing classes with a percentage lower than the given threshold
        # threshold, x_train, t_train = self.clean_data(
        #     self.min_sample_label, x_train, t_train)

        # Encode the target variables as one-hot vectors
        # t_train = self.one_hot_vector_encoding(data=t_train)
        # t_val = self.one_hot_vector_encoding(data=t_val)

        return data, x_train, x_val, t_train, t_val

    def one_hot_vector_encoding(self, data: np.ndarray) -> np.ndarray:
        """
        Encodes target values into one-hot vectors.

        Args:
        - data (np.ndarray): array of target values.

        Returns:
        - np.ndarray: array of one-hot encoded target values.
        """
        # Create a OneHotEncoder object and fit it to the target data
        encoder = OneHotEncoder(categories=[range(99)])
        # Reshape the target data to 2D array and transform it to one-hot vectors
        return encoder.fit_transform(np.array(data).reshape(-1, 1)).toarray()

    def clean_data(self, threshold, x, t):
        """
        Clean the data by removing classes with a percentage lower than the given threshold.

        Args:
            threshold (int): The threshold value.
            x (numpy.ndarray): The input data.
            t (numpy.ndarray): The target data.

        Returns:
            tuple: A tuple containing the threshold value, the cleaned input data, and the cleaned target data.
        """
        # Initialize a dictionary to count the number of occurrences of each class in the target data
        class_count = {}

        # Count the number of occurrences of each class in the target data.
        for i, target in enumerate(t):
            if target not in class_count:
                class_count[target] = {'count': 1, 'indices': [i]}
            else:
                class_count[target]['count'] += 1
                class_count[target]['indices'].append(i)

        # Create a list of indices to remove based on the classes with low occurrence percentage
        data_to_delete = []
        class_counts = []
        for key in class_count:
            class_percentage = class_count[key]['count']
            class_counts.append(class_count[key]['count'])
            if class_percentage < threshold:
                data_to_delete += class_count[key]['indices']

        # Remove data with classes having a percentage lower than the given threshold.
        # If the percentage of data to be deleted is greater than 10%, do not delete any data
        # and update the threshold value to the smallest class count
        if (len(data_to_delete)/len(x)) < 0.10:
            x = np.delete(x, data_to_delete, axis=0)
            t = np.delete(t, data_to_delete)
        else:
            threshold = np.min(class_counts)

        return threshold, x, t
    