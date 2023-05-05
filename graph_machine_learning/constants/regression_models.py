from enum import Enum


class RegressionModels(Enum):
    LASO = 1
    SVR = 2
    DECISION_TREE = 3
    ELASTIC_NET = 4
    RANDOM_FOREST = 5
    RIDGE = 6
