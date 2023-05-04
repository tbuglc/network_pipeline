from sklearn.linear_model import ElasticNet
import numpy as np

from models.base.base_regression import BaseRegression

class ElasticNetRegression(BaseRegression):
    def __init__(self) -> None:
        super().__init__()
        self.model = ElasticNet(alpha=1.0, l1_ratio=0.5)