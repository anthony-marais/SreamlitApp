import streamlit as st  # pip install streamlit
import pandas as pd  # pip install pandas openpyxl
import joblib


def forms():

    
    model = open("./models/model.pkl", "rb")
    lgbm_model = joblib.load(model)

    values = []

    ParameterInputValue = []

    C1, C2, C3 = st.columns(3)
    with C1:
        InflightWifiService = st.slider("Inflight Wifi Service",value=1,min_value=1, max_value=5, step=1)
        ParameterInputValue.append(InflightWifiService)

    with C2:
        EaseOfOnlineBooking = st.slider("Ease Of Online Booking",value=1,min_value=1, max_value=5, step=1)
        ParameterInputValue.append(EaseOfOnlineBooking)

    with C3:
        FoodAndDrink = st.slider("Food And Drink",value=1,min_value=1, max_value=5, step=1)
        ParameterInputValue.append(FoodAndDrink)
    C4,C5,C6 = st.columns(3)
    
    with C4:
        OnlineBoarding = st.slider("Online Boarding",value=1,min_value=0, max_value=5, step=1)
        ParameterInputValue.append(OnlineBoarding)

    with C5:
        SeatComfort = st.slider("Seat Comfort",value=1,min_value=0, max_value=5, step=1)            
        ParameterInputValue.append(SeatComfort)
    
    with C6:
        InflightEntertainment = st.slider("Inflight Entertainment",value=1,min_value=0, max_value=5, step=1)
        ParameterInputValue.append(InflightEntertainment)

    C7,C8,C9 = st.columns(3)
    
    with C7:
        OnboardService = st.slider("On-board Service",value=1,min_value=0, max_value=5, step=1)
        ParameterInputValue.append(OnboardService)

    with C8:
        LegRoom = st.slider("Leg Room",value=1,min_value=1, max_value=5, step=1)      
        ParameterInputValue.append(LegRoom)
      
    with C9:
        BaggageHandling = st.slider("Baggage Handling",value=1,min_value=0, max_value=5, step=1)
        ParameterInputValue.append(BaggageHandling)

    C10,C11,C12 = st.columns(3)
    
    with C10:
        CheckinService = st.slider("Checkin Service",value=1,min_value=0, max_value=5, step=1)
        ParameterInputValue.append(CheckinService)

    with C11:
        InflightService = st.slider("Inflight Service",value=1,min_value=0, max_value=5, step=1)            
        ParameterInputValue.append(InflightService)

    with C12:
        Cleanliness = st.slider("Cleanliness",value=1,min_value=1, max_value=5, step=1)
        ParameterInputValue.append(Cleanliness)

    C13,C14,C15 = st.columns(3)
    
    with C13:
        ClassEconomy = st.slider("Class_Economy",value=1,min_value=0, max_value=1, step=1)
        ParameterInputValue.append(ClassEconomy)

    
    with C14:
        CustomerTypeReturningCustomer = st.slider("Customer Type_Returning Customer",value=1,min_value=0, max_value=1, step=1)            
        ParameterInputValue.append(CustomerTypeReturningCustomer)

    
    with C15:
        TypeOfTravelPersonalTravel = st.slider("Type Of Travel_Personal Travel",value=1,min_value=0, max_value=1, step=1)
        ParameterInputValue.append(TypeOfTravelPersonalTravel)

    
    ParameterList1_5 = ["Inflight Wifi Service", "Ease Of Online Booking", "Food And Drink",
                            "Online Boarding", "Seat Comfort", "Inflight Entertainment", 
                            "On-board Service", "Leg Room", "Baggage Handling", 
                            "Checkin Service", "Inflight Service", "Cleanliness"
                            ]

    ParameterList0_1 = ["Class_Economy", "Customer Type_Returning Customer", "Type Of Travel_Personal Travel"]

    ParameterList = [y for x in [ParameterList1_5,ParameterList0_1]for y in x]

    input_variables=pd.DataFrame([ParameterInputValue],columns=ParameterList,dtype=float)

    #return input_variables


    st.dataframe(input_variables)
            
    st.write('\n\n')

    if st.button("Click Here to Classify"):
        
        if lgbm_model.predict(input_variables) == 1:
            st.subheader("😃👍")
            st.success("The Flight is a Good Flight, with a score of: {:2.2%}".format(lgbm_model.predict_proba(input_variables)[0][1]))
        else:
            st.subheader("😢👎")
            st.error("The Flight is a Bad Flight with a score of: {:2.2%}".format(lgbm_model.predict_proba(input_variables)[0][0]))  


        #prediction = lgbm_model.predict(input_variables)
        #sentiment = {'eloignement_Probability': prediction[0, 1]}
        #st.write(prediction)