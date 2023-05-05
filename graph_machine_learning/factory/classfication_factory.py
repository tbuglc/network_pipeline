from constants.classification_models import ClassificationModels
from models.classification.k_neast_neighbors import KNN
from models.classification.support_vector_machine import SVM
from models.classification.decision_tree import DecisionTree
from models.classification.simple_perceptron import SimplePerceptron
from models.classification.random_forest import RandomForest
# from models.base.base_factory import BaseFactory


class ClassificationFactory():
  # A static method that takes an integer `model` as input and returns a machine learning model based on the input value.
  @staticmethod
  def get_model(self,model=int):
    # Check the value of the input `model` and return the corresponding machine learning model.
    if model == ClassificationModels.KNN:
           print('KNN Model')
           return KNN()
    if model == ClassificationModels.SVM:
           print('SVM Model')
           return SVM()
    if model == ClassificationModels.DECISION_TREE:
           print('Decision Tree Model')
           return DecisionTree()
    if model == ClassificationModels.SIMPLE_PERCEPTRON:
           print('Perceptron Model')
           return SimplePerceptron()
    if model == ClassificationModels.RANDOM_FOREST:
           print('Random Forest')
           return RandomForest()
 
    # Return None if no matching model is found.
    print('== No model found ==')
    return None
