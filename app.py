import pandas as pd
import numpy as np
import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error,mean_absolute_error,r2_score
import warnings
warnings.filterwarnings("ignore")

st.title("Ames Housing Price Prediction App")
st.write("choose a model to predict the house price in ames city limits")

@st.cache_data
def load_data():
    df=pd.read_csv(r"C:\Users\punit\Downloads\Ames_Housing_Subset.csv")
    return df

df1=load_data()

st.subheader("Dataset preview")
st.dataframe(df1)

df1=df1.select_dtypes(include=[np.number])
X=df1.drop(["SalePrice"],axis=1)
y=df1["SalePrice"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model_name=st.selectbox("select the model",["Linear Regression","Decision Tree","Random Forest Regressor"])
if model_name=="Linear Regression":
    model = LinearRegression()
elif model_name=="Decision Tree":
    model = DecisionTreeRegressor()
else:
    model = RandomForestRegressor()

model.fit(X_train,y_train)

y_pred=model.predict(X_test)
mse=mean_squared_error(y_test,y_pred)
mae=mean_absolute_error(y_test,y_pred)
r2=r2_score(y_test,y_pred)

st.subheader("mse")
st.write(mse)
st.subheader("mae")
st.write(mae)
st.subheader("r2_score")
st.write(r2)

#input values
st.subheader("enter the input values")
input_data={}
for col in X.columns:
    input_data[col]=st.number_input(f"Enter{col}",
                                    value=X[col].median())

input_df=pd.DataFrame([input_data])

if st.button("Predict Price"):
    price=model.predict(input_df)[0]
    st.success(f"Predicted Price is ${price:.2f}")
