import streamlit as st
import numpy as np
import pickle

def load_data():
    with open('saved_steps.pk1','rb') as file:
        data= pickle.load(file)
    return data

data= load_data()

regressor= data['model']
le_country=data['le_country']
le_education=data['le_education']

def show_predict_page():
    st.title("Software Developer Salary Prediction")
    st.write("""### we need some information to predict the salary""")

    countries={
        "United States",
        "India",  
        "United Kingdom",      
        "Germany",             
        "Canada",             
        "Brazil",                
        "France",               
        "Spain",              
        "Australia",             
        "Netherlands",         
        "Poland",               
        "Italy",                 
        "Russian Federation",  
        "Sweden"                
    }

    education={
        "Less than a Bachelors",
        "Bachelor’s degree", 
        "Master’s degree", 
        "Post grad"

    }
  
    
    

    country=st.selectbox("Country",countries)
    education=st.selectbox("Education Level",education)
    experience= st.slider("Years of Experience",0,50,3)
    ok=st.button("Calculate Salary")
    if ok:
        x= np.array([[country,education,experience]])
        x[:,0]= le_country.transform(x[:,0])
        x[:,1]= le_education.transform(x[:,1])
        x= x.astype(float)

        salary= regressor.predict(x)
        st.subheader(f"The Estimated Salary is ${salary[0]:.2f}")

