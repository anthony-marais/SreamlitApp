import pandas as pd
from string import capwords


def clean_data(df):
    """
    This function cleans the dataframe.
    :param df: dataframe
    :return: dataframe
    - Column names are renamed.
    - Elements in Features 'Customer Type' and 'Class' are renamed.
    - Rows with Null values are removed.
    - Rows with scores of 0 in the survey of satisfaction are removed (Customers probably did not indicate).
    - Departure Delay and Arrival Delay are combined.
    - Satisfaction target is relabelled as 0 and 1.
    """
    df['Customer Type'] = df['Customer Type'].map(
        {'Loyal Customer': 'Returning Customer', 'disloyal Customer': 'First-time Customer'})
    df = df.dropna(axis=0)
    df['Departure Delay in Minutes'] = df['Departure Delay in Minutes'].astype(
        'float')
    df = df.rename(columns={'Leg room service': 'Leg room'})
    df.columns = [capwords(i) for i in df.columns]
    df = df.rename(columns={
                   'Departure/arrival Time Convenient': 'Departure/Arrival Time Convenience'})
    df = df[(df['Inflight Wifi Service'] != 0) & (df['Departure/Arrival Time Convenience'] != 0) & (df['Ease Of Online Booking'] != 0) & (df['Gate Location']) & (df['Food And Drink'] != 0) & (df['Online Boarding'] != 0) & (df['Seat Comfort']
                                                                                                                                                                                                                               != 0) & (df['Inflight Entertainment'] != 0) & (df['On-board Service'] != 0) & (df['Leg Room'] != 0) & (df['Baggage Handling'] != 0) & (df['Checkin Service'] != 0) & (df['Inflight Service'] != 0) & (df['Cleanliness'] != 0)]
    df['Satisfaction'] = df['Satisfaction'].map(
        {'satisfied': 1, 'neutral or dissatisfied': 0})
    df = df.reset_index()
    df = df.drop('index', axis=1)
    df['Total Delay'] = df['Departure Delay In Minutes'] + \
        df['Arrival Delay In Minutes']
    DF = df.copy()
    df = df.drop('Id', axis=1)
    df = df.reindex(columns=['Satisfaction'] +
                    list(df.columns)[:-2]+['Total Delay'])
    df = df.drop(['Departure Delay In Minutes',
                 'Arrival Delay In Minutes'], axis=1)
    df['Class'] = df['Class'].map(
        {'Eco': 'Economy', 'Eco Plus': 'Economy', 'Business': 'Business'})

    return df

def standardize_data(df1):
    """
    This function standardizes the dataframe.
    """
    df2 = pd.get_dummies(df1,columns=['Gender','Customer Type','Type Of Travel','Class'],drop_first=True)
    df2 = df2.drop('Gender_Male',axis=1)
    df2 = df2.drop(['Total Delay', 'Flight Distance', 'Age',
               'Gate Location', 'Departure/Arrival Time Convenience'], axis=1)

    return df2