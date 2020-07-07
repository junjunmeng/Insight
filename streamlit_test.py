import numpy as np
import pandas as pd
import streamlit as st
from sklearn.linear_model import LogisticRegression


### build the model

LR =LogisticRegression(solver = 'liblinear')

X_res = pd.read_csv('./Melanoma/X_res.csv', index_col= 0)
y_res = pd.read_csv('./Melanoma/y_res.csv', index_col= 0)

LR.fit(X_res, y_res)

data = np.array([76, 11, 2, 0, 0, 0, ])
def Model_Batch(data):
  risk = LR.predict_proba(data)
  return risk[:,0]

# st.title("Melanoma Forecaster")
