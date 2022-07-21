import streamlit as st
import pandas as pd
import yaml
from time import sleep
import streamlit_authenticator as stauth # pip install streamlit-authenticator

PAGE_TITLE = 'Home'


def write():

   with open("./config.yml") as file:
            config = yaml.load(file, Loader=yaml.SafeLoader)
    
   authenticator = stauth.Authenticate(
            config["credentials"],
            config["cookie"]["name"],
            config["cookie"]["key"],
            config["cookie"]["expiry_days"],
            config["preauthorized"],
        )

   st.markdown(f'# {PAGE_TITLE}')
    
   st.markdown("Hello World!")

   with open("./config.yml") as file:
            config = yaml.load(file, Loader=yaml.SafeLoader)

    

   st.sidebar.success("Welcome back, {}".format(st.session_state.name))

    #menu = ["Home", "About", "Contact", "Logout"]
    #choice = st.sidebar.selectbox("Menu",menu)
    #if choice == "Home":
    #    st.subheader("Home")

   authenticator.logout("Logout", "sidebar")


if __name__=='__main__':
    write()