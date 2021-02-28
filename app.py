import streamlit as st
import pandas as pd
import pickle
import numpy as np
from sklearn import preprocessing
model = pickle.load(open('my_model.pkl','rb'))

df = pd.read_csv('features_data.csv')

# initializing encoders for each column
le_target = preprocessing.LabelEncoder()
le_country = preprocessing.LabelEncoder() 
le_city = preprocessing.LabelEncoder() 
le_plan = preprocessing.LabelEncoder() 
# getting the labels in columns  
le_target.fit(df["target"])
le_country.fit(df["country"])
le_city.fit(df["city"])
le_plan.fit(df["plan"])


def predict_churn(birth_year,country,city,plan,User_gain_usd, num_contacts,Transaction_per_user,notification_received):
    input=np.array([[birth_year,country,city,plan, User_gain_usd , num_contacts,Transaction_per_user,notification_received]]).astype(np.float64)
    prediction=model.predict_proba(input)
    pred='{0:.{1}f}'.format(prediction[0][0], 2)
    return float(pred)

def main():
    st.title("Real-time churn user prediction")
    html_temp = """
    <div style="background-color:#025246 ;padding:10px">
    <h4 style="color:white;text-align:center;"> User Informations </h4>
    </div>
    """
    st.markdown(html_temp, unsafe_allow_html=True)

    Transaction_per_user = st.text_input("NbrTransactions","Type Here")
    num_contacts = st.text_input("NbrContacts","Type Here")
    country = st.text_input("Country","Type Here")
    city = st.text_input("City","Type Here")
    plan = st.text_input("Plan","Type Here")
    birth_year = st.text_input("BirthYear","Type Here")
    notification_received = st.text_input("NotificationsReceived","Type Here")
    User_gain_usd = st.text_input("MoneyRaised","Type Here")
    ENGAGED_html="""  
      <div style="background-color:#F4D03F;padding:10px >
       <h2 style="color:white;text-align:center;"> User is ENGAGED </h2>
       </div>
    """
    CHURN_html="""  
      <div style="background-color:#F08080;padding:10px >
       <h2 style="color:black ;text-align:center;"> User is CHURNED</h2>
       </div>
    """

    if st.button("Predict"):
        output=predict_churn(birth_year,le_country.transform([country])[0], le_city.transform([city])[0] ,le_plan.transform([plan])[0],User_gain_usd, num_contacts,Transaction_per_user,notification_received)

        if output >= 0.5:
            st.markdown(ENGAGED_html,unsafe_allow_html=True)
        else:
            st.markdown(CHURN_html,unsafe_allow_html=True)

if __name__=='__main__':
    main()