from secrets import choice
import pandas as pd  # pip install pandas openpyxl
import joblib
# import plotly.express as px  # pip install plotly-express
import streamlit as st  # pip install streamlit
import streamlit_authenticator as stauth # pip install streamlit-authenticator

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
       menu = ["Home","Classification","About", "Contact"]
       choice = st.sidebar.selectbox("Menu",menu)
       authenticator.logout("Logout", "sidebar")
       
       if choice == "Home":
            st.title("Home")
            st.subheader("Welcome to Airline Customer Satisfaction Rating App ✈️!")
            st.markdown("This is a Streamlit app that allows you to rate your airline customer satisfaction.")
            st.markdown("You can rate your airline customer satisfaction by selecting a classification models from a dropdown menu.")
       
       if choice == "Classification":
            st.title("Classification App")
            model = open("./models/model.pkl", "rb")
            lgbm_model = joblib.load(model)
            

            ParameterList1_5 = ["Inflight Wifi Service", "Ease Of Online Booking", "Food And Drink",
                            "Online Boarding", "Seat Comfort", "Inflight Entertainment", 
                            "On-board Service", "Leg Room", "Baggage Handling", 
                            "Checkin Service", "Inflight Service", "Cleanliness"
                            ]

            ParameterList0_1 = ["Class_Economy", "Customer Type_Returning Customer", "Type Of Travel_Personal Travel"]

            ParameterInputValue = []

            ParameterDefaultValue1_5 = ["1","2","3","4","5","1","2","3","4","5","1","2","3","4","5"]
            ParameterDefaultValue0_1 = ["0","1","1"]

            values=[]

            for parameter,parameter_df in zip(ParameterList1_5,ParameterDefaultValue1_5):
                values = st.slider(label=parameter, key=parameter,value=float(parameter_df),min_value=1.0, max_value=5.0, step=1.0)
                ParameterInputValue.append(values)

            for parameter,parameter_df in zip(ParameterList0_1,ParameterDefaultValue0_1):
                values = st.slider(label=parameter, key=parameter,value=float(parameter_df),min_value=0.0, max_value=1.0, step=1.0)

                ParameterInputValue.append(values)

            ParameterList = [y for x in [ParameterList1_5,ParameterList0_1]for y in x]


            input_variables=pd.DataFrame([ParameterInputValue],columns=ParameterList,dtype=float)

            st.dataframe(input_variables)
            
            st.write('\n\n')

            if st.button("Click Here to Classify"):
                
                prediction = lgbm_model.predict_proba(input_variables)

                st.write(prediction)

       if choice == "About":
            c1, c2, c3 = st.columns(3)
            with c1:
                initialInvestment = st.slider("Starting capital",value=1,min_value=0, max_value=1, step=1)
            with c2:
                monthlyContribution = st.text_input("Monthly contribution (Optional)",value=100)
            with c3:
                annualRate = st.text_input("Annual increase rate in percentage",value="15")
           
            C4,C5,C6 = st.columns(3)
            
            with C4:
                years = st.slider("Number of years",value=1,min_value=0, max_value=5, step=1)
            with C5:
                yearss = st.slider("Nusmber of years",value=1,min_value=0, max_value=5, step=1)            
            with C6:
                yesars = st.slider("Numbesr of years",value=1,min_value=0, max_value=5, step=1)



    with open('config.yml', 'w') as file:
        yaml.dump(config, file, default_flow_style=False)



if __name__ == '__main__':
	main()




