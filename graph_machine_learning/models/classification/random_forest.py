import numpy as np
from sklearn.ensemble import RandomForestClassifier
from models.base.base_classfication import BaseClassification



class RandomForest(BaseClassification):
    def __init__(self) -> None:
        super().__init__()
        self.model = RandomForestClassifier()
    # override 
    def train(self, x_train, x_val, t_train, t_val):
        # Train the classifier on the training set
        t_train = np.argmax(t_train, axis=1)
        t_val = np.argmax(t_val, axis=1)
        
        return super().train(x_train=x_train, x_val=x_val, t_train=t_train,t_val=t_val)
     