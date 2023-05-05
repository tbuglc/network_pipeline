from sklearn.neighbors import KNeighborsClassifier
from models.base.base_classfication import BaseClassification


class KNN(BaseClassification):
    def __init__(self) -> None:
        super().__init__() # initialize base class attributes
        self.model = KNeighborsClassifier() # override the parent self.model value