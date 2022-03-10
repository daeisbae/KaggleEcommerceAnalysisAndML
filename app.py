import streamlit as st
import pandas as pd
from joblib import load
import numpy as np

st.write(
    """
    ## Ecommerce Data
    """
)

#Load dataset
eco = pd.read_csv("EcommerceCustomers.csv")

#Load linear regression trained model
model = load('ecommerce_model_sklearn.joblib')

##Prediction bar
def MakeControlBar():
    st.sidebar.subheader('Expected Yearly Amount Spent')
    Session = st.sidebar.slider('Avg Session Length (Minutes)', float(eco['Avg. Session Length'].min()), float(eco['Avg. Session Length'].max()), float(eco['Avg. Session Length'].mean()))
    App = st.sidebar.slider('Time on App (Minutes)', float(eco['Time on App'].min()), float(eco['Time on App'].max()), float(eco['Time on App'].mean()))
    Membership = st.sidebar.slider('Length of Membership (Years)', float(eco['Length of Membership'].min()), float(eco['Length of Membership'].max()), float(eco['Length of Membership'].mean()))

    return Session, App, Membership

Controlbar = MakeControlBar()

st.markdown('### Dataset')
st.dataframe(eco, width=1000)
st.download_button('Download This Dataset!', data=eco.to_csv().encode('utf-8'), file_name='Ecommerce_Dataset.csv')

TopCustomer = eco[['Email', 'Yearly Amount Spent']].set_index('Email')\
    .sort_values('Yearly Amount Spent', ascending=False).head(10)

st.write(
    """
    ### top 10 vip customers
    """
)

#Bar chart for top 10 customers
st.bar_chart(TopCustomer, height=800)

#Model prediction
st.markdown('### Model Prediction Result')
st.write(f"Expected Expenditure: $ {model.predict(np.array([[*Controlbar]]))[0]:.2f}")