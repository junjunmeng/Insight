import numpy as np
from sklearn.linear_model import LogisticRegression
import pandas as pd

LR =LogisticRegression(C=1.0, class_weight=None, dual=False, fit_intercept=True,
                   intercept_scaling=1, l1_ratio=None, max_iter=100,
                   multi_class='auto', n_jobs=None, penalty='l2',
                   random_state=None, solver='liblinear', tol=0.0001, verbose=0,
                   warm_start=False)

X_res = pd.read_csv('./Melanoma/X_res.csv', index_col= 0)
y_res = pd.read_csv('./Melanoma/y_res.csv', index_col= 0)

LR.fit(X_res, y_res)

demo_features = ['RIDAGEYR','edit-tumor-location', 'RIAGENDR', 'NUMMAT','THICKNESS', 'ulceration']


for i in demo_features:
      tmp = request.args.get(i)
      print(tmp)

def get_data():
    #pull 'data' from input field and store it
    global data
    data_orginal = []
    for i in demo_features:
      tmp = request.args.get(i)
      data_orginal.append(tmp)
    data_str = []
    for item in data_orginal:
      data_str.extend(item.split(","))
    data = [float(i) for i in data_str]
    return data
data = get_data()
print (data)

@app.route('/Predict')
def predict():
    global risk
    data = get_data()
    risk = Model_Batch(data)
    return render_template('Predict.html', result = risk)

