import streamlit as st
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier

@st.cache_data
def load_data():
    iris=load_iris()
    df=pd.DataFrame(iris.data, columns=iris.feature_names)
    df['species']=iris.target
    return df,iris.target_names

df, target_names=load_data()

model=RandomForestClassifier()
model.fit(df.iloc[:,:-1],df['species'])

st.sidebar.title("Input Features")
#Sepal_length=st.slider("Sepal length",float(['sepel_length(cm)'].min()),float(['sepel_length(cm)'].max()))
#Sepal_width=st.slider("Sepal width",float(['sepel_width(cm)'].min()),float(['sepel_width(cm)'].max()))
#Petal_length=st.slider("Petal length",float(['petal_length(cm)'].min()),float(['petal_length(cm)'].max()))
#Petal_width=st.slider("Petal width",float(['petal_width(cm)'].min()),float(['petal_width(cm)'].max()))


Sepal_length = st.slider(
    "Sepal length",
    float(df['sepal length (cm)'].min()),
    float(df['sepal length (cm)'].max())
)

Sepal_width = st.slider(
    "Sepal width",
    float(df['sepal width (cm)'].min()),
    float(df['sepal width (cm)'].max())
)

Petal_length = st.slider(
    "Petal length",
    float(df['petal length (cm)'].min()),
    float(df['petal length (cm)'].max())
)

Petal_width = st.slider(
    "Petal width",
    float(df['petal width (cm)'].min()),
    float(df['petal width (cm)'].max())
)

input_data=[[Sepal_length, Sepal_width, Petal_length, Petal_width]]

##Pridiction
prediction= model.predict(input_data)
predicted_species=target_names[prediction[0]]

st.write("Prdict")
st.write(f" The Predicted Species is : {predicted_species}")