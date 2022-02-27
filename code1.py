import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression  
from sklearn.ensemble import RandomForestClassifier

# Loading the dataset.
iris_df = pd.read_csv("iris-species.csv")

# Adding a column in the Iris DataFrame to resemble the non-numeric 'Species' column as numeric using the 'map()' function.
# Creating the numeric target column 'Label' to 'iris_df' using the 'map()' function.
iris_df['Label'] = iris_df['Species'].map({'Iris-setosa': 0, 'Iris-virginica': 1, 'Iris-versicolor':2})

# Creating a model for Support Vector classification to classify the flower types into labels '0', '1', and '2'.

# Creating features and target DataFrames.
X = iris_df[['SepalLengthCm','SepalWidthCm', 'PetalLengthCm', 'PetalWidthCm']]
y = iris_df['Label']

# Splitting the data into training and testing sets.
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.33, random_state = 42)

# Creating the SVC model and storing the accuracy score in a variable 'score'.
svc_model = SVC(kernel = 'linear')
svc_model.fit(X_train, y_train)
#score = svc_model.score(X_train, y_train)
rfc = RandomForestClassifier(n_jobs=-1,n_estimators=50)
rfc.fit(X_train,y_train)

lr = LogisticRegression()
lr.fit(X_train,y_train)

def prediction(model, sepal_length, sepal_width, petal_length, petal_width):
  species = model.predict([[sepal_length, sepal_width, petal_length, petal_width]])
  species = species[0]
  if species == 0:
    return "Iris-setosa"
  elif species == 1:
    return "Iris-virginica"
  else:
    return "Iris-versicolor"  


st.sidebar.title('Iris Flower Prediction App')
s_len=st.sidebar.slider('Sepal Length',float(iris_df['SepalLengthCm'].min()),float(iris_df['SepalLengthCm'].max()))
s_wid = st.sidebar.slider("Sepal Width", float(iris_df["SepalWidthCm"].min()), float(iris_df["SepalWidthCm"].max()))
p_len = st.sidebar.slider("Petal Length", float(iris_df["PetalLengthCm"].min()), float(iris_df["PetalLengthCm"].max()))
p_wid = st.sidebar.slider("Petal Width", float(iris_df["PetalWidthCm"].min()), float(iris_df["PetalWidthCm"].max()))

model = st.sidebar.selectbox('Model',('SVM','RFC','LR'))
if st.sidebar.button('Predict'):
	if model == 'SVM':
		species=prediction(svc_model,s_len,s_wid,p_len,p_wid)
		score = svc_model.score(X_train,y_train)
	if model == 'RFC':
		species=prediction(rfc,s_len,s_wid,p_len,p_wid)
		score = rfc.score(X_train,y_train)
	if model == 'LR':
		species=prediction(lr,s_len,s_wid,p_len,p_wid)
		score = lr.score(X_train,y_train)
st.write('Species predicted is ',species)
st.write('Accuracy of ',score)