import numpy as np
from sklearn.linear_model import LogisticRegression
# import pandas as pd
import pickle

X= './Melanoma/X_ref.pkl'
Y = './Melanoma/y_ref.pkl'

LR =LogisticRegression(solver = 'liblinear')

with open(X,'rb') as file:
  X_res = pickle.load(file)

with open(Y, 'rb') as file:
  y_res = pickle.load(file)

LR.fit(X_res, y_res)

def Model_One(data):
  risk = LR.predict_proba(data.reshape(1,-1))
  v = round(risk[0][1],3)
  return v




