import pickle
import pandas as pd
import numpy as np

class species_predictor():
  def __init__(self):
    pass
  
  def deserialize(self):
    # de-serialize mlp_nn.pkl file into an object called model using pickle
    with open('classifier.pkl', 'rb') as handle:
      model = pickle.load(handle)
    return model
  
  def predict(self, sepal_length,sepal_width,petal_length,petal_width):
    model = self.deserialize()
    return model.predict(np.array([[sepal_length,sepal_width,petal_length,petal_width]]))