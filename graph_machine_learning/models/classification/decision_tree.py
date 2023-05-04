from sklearn import tree
from models.base.base_classfication import BaseClassification

class DecisionTree(BaseClassification):
    def __init__(self) -> None:
        super().__init__()
        self.model = tree.DecisionTreeClassifier()
        