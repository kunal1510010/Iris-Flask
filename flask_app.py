import json
import os
from flask import Flask,jsonify,request
from flask_cors import CORS
from Model import species_predictor
import pickle
import numpy as np
from sklearn.neighbors import KNeighborsClassifier

app = Flask(__name__)
CORS(app)
@app.route("/price/", methods=['GET'])
def return_price():
  sepal_length = float(request.args.get('slength'))
  sepal_width = float(request.args.get('swidth'))
  petal_length = float(request.args.get('plength'))
  petal_width = float(request.args.get('pwidth'))


  with open('classifier.pkl', 'rb') as handle:
  	model = pickle.load(handle)
    
  return "Output" + str(model.predict(np.array([[sepal_length,sepal_width,petal_length,petal_width]])))
  # species = species_predictor.predict(self,sepal_length,sepal_width,petal_length,petal_width) 
  # species_dict = {
  #               'model':'mlp',
  #               'species': species,
  #               }
# methods=['GET']
  # return jsonify(species_dict)

@app.route("/",methods=['GET'])
def default():
  return "<h1> Welcome to Iris species predictor <h1>"

if __name__ == "__main__":
    app.run() 

    