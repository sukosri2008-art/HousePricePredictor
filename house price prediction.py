import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score,confusion_matrix
import streamlit as st
from sklearn.linear_model import LinearRegression
data=pd.read_csv(r"C:\Users\srija\OneDrive\Documents\house prediction.csv")
df=pd.DataFrame(data)
df=df.dropna()
X=df.drop("price",axis=1)
y=df["price"]
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=42)
model=LinearRegression()
model.fit(X_train,y_train)
st.title("HOUSE PRICE PREDICTION")
st.subheader("Predicting the House price with the features")
st.write("Fill the below details completely")
n=st.text_input("House Owner Name:")
st.number_input("Price",min_value=10000,max_value=10000000)
st.slider("Bedroom:",min_value=1,max_value=20,value=3)
st.slider("Bathroom:",min_value=1,max_value=20,value=3)
st.number_input("Sq.ft_living",min_value=1000,max_value=20000)
st.number_input("Sq.ft_lot",min_value=1000,max_value=100000000)
st.slider("Floor:",min_value=1,max_value=10,value=3)
st.selectbox("WaterFront",[0,1])
st.slider("View:",min_value=1,max_value=10,value=0)
st.slider("Condition:",min_value=1,max_value=10,value=3)
if st.button("Predict"):
    pred=model.predict(X_test)
    st.write("Prediction:",pred)
    if pred[0] > 1000000:
         st.success(f"{n} has bought a luxurious house")
         st.balloons()
    else:
          st.error(f"{n} has bought a decent living  house")
        


