from secrets import choice
import pandas as pd  # pip install pandas openpyxl
import joblib
# import plotly.express as px  # pip install plotly-express
import streamlit as st  # pip install streamlit
import streamlit_authenticator as stauth # pip install streamlit-authenticator
from forms_classifier import forms
import yaml  # pip install pyaml


st.set_page_config(
    page_title="Streamlit App",
    page_icon="✈️",
    layout="centered",
    menu_items={
        "Get Help": "https://www.extremelycoolapp.com/help",
        "Report a bug": "https://www.extremelycoolapp.com/bug",
        "About": "# This is a header. This is an *extremely* cool app!",
    
    },
)



def main():

    
    with open("./config.yml") as file:
            config = yaml.load(file, Loader=yaml.SafeLoader)
    
    authenticator = stauth.Authenticate(
            config["credentials"],
            config["cookie"]["name"],
            config["cookie"]["key"],
            config["cookie"]["expiry_days"],
            config["preauthorized"],
        )


    
    
    name, authentication_status, username = authenticator.login("Login", "main")


    if authentication_status == False:
        st.error("Username/password is incorrect")
        if authenticator.register_user('Register user', preauthorization=False):
            st.success('User registered successfully')
    if authentication_status == None:
        st.warning("Please enter your username and password")
        if authenticator.register_user('Register user', preauthorization=False):
            st.success('User registered successfully')


    if authentication_status:
       
       st.sidebar.success("Welcome back, {}".format(st.session_state.name))
       menu = ["Home","Classification"]
       choice = st.sidebar.selectbox("Menu",menu)
       authenticator.logout("Logout", "sidebar")
       
       if choice == "Home":
            st.title("Home")
            st.subheader("Welcome to Airline Customer Satisfaction Rating App ✈️!")
            st.markdown("This is a Streamlit app that allows you to rate your airline customer satisfaction.")
            st.markdown("You can rate your airline customer satisfaction by selecting a classification models from a dropdown menu.")
       
       if choice == "Classification":
            st.title("Classification App")
            forms()




    with open('config.yml', 'w') as file:
        yaml.dump(config, file, default_flow_style=False)



if __name__ == '__main__':
	main()




