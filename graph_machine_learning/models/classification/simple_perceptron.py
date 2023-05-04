import numpy as np
from sklearn.linear_model import Perceptron
from models.base.base_classfication import BaseClassification


class SimplePerceptron(BaseClassification):
    def __init__(self) -> None:
        super().__init__()
        self.model = Perceptron()

    def train(self, x_train, x_val, t_train, t_val):
        t_train = np.argmax(t_train, axis=1)
        t_val = np.argmax(t_val, axis=1)
        
        return super().train(x_train=x_train, x_val=x_val, t_train=t_train,t_val=t_val)