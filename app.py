import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import streamlit as st

# load data
data = pd.read_csv('creditcard.csv')

# separate legitimate and fraudulent transactions
legit = data[data.Class == 0]
fraud = data[data.Class == 1]

# undersample legitimate transactions to balance the classes
legit_sample = legit.sample(n=len(fraud), random_state=2)
data = pd.concat([legit_sample, fraud], axis=0)

# split data into training and testing sets
X = data.drop(columns="Class", axis=1)
y = data["Class"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=2)

# train logistic regression model
model = LogisticRegression()
model.fit(X_train, y_train)

# evaluate model performance
train_acc = accuracy_score(model.predict(X_train), y_train)
test_acc = accuracy_score(model.predict(X_test), y_test)

# create Streamlit app
st.title("Credit Card Fraud Detection Model")
st.write("Enter the Features below to check if the Transaction is Legit or Fraud:")

# create input fields for user to enter feature values
input_df = st.text_input('Input All features')
input_df_lst = input_df.split(',')
# create a button to submit input and get prediction
submit = st.button("Submit")

if submit:
    # get input feature values
    input_df_lst = [value.strip('"') for value in input_df_lst]
    # while len(input_df_lst) < 30:
    #     input_df_lst.append(0.0)
    features = np.array(input_df_lst, dtype=np.float64)
    # try:
    #     features = np.array([float(value) for value in input_df_lst], dtype=np.float64)
    # except ValueError as e:
    #     st.error(f"Invalid input detected: {e}")
    # features = np.array(input_df_lst, dtype=np.float64)
    # make prediction
    # feature_names = ["Time","V1","V2","V3","V4","V5","V6","V7","V8","V9","V10","V11","V12","V13","V14","V15","V16","V17","V18","V19","V20","V21","V22","V23","V24","V25","V26","V27","V28","Amount","Class"]
    # features_df = pd.DataFrame([features], columns=feature_names)
    # prediction = model.predict(features_df)
    prediction = model.predict(features.reshape(1, -1))
    # display result
    if prediction[0] == 0:
        st.write("Legitimate Transaction")
    else:
        st.write("Fraudulent Transaction")