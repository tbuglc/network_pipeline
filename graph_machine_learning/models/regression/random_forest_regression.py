from sklearn.ensemble import RandomForestRegressor
import numpy as np
from models.base.base_regression import BaseRegression

class RandomForestRegression(BaseRegression):
    def __init__(self) -> None:
        super().__init__()
        self.model = RandomForestRegressor()