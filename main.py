import streamlit as st
import pandas as pd 
from sklearn.metrics import r2_score,mean_squared_error
from sklearn.model_selection import train_test_split
from PIL import Image
from sklearn.linear_model import LinearRegression
from sklearn import linear_model
st.write("""
Student Final Marks Prediction Using Regression

""")
image=Image.open("C:/Users/SARVJEET/Desktop/Post/ml.jpeg")
st.image(image,caption="Machine Learning", use_column_width=True)
df=pd.read_csv('C:/Users/SARVJEET/MY Project/student-por.csv')
data=df[["studytime","freetime","G2","G1","absences","health","G3"]]
st.subheader('Data Information: ')
st.dataframe(data)
st.write(data.describe())

chart=st.bar_chart(data)

X=data.iloc[:,0:6].values
y=data.iloc[:,-1].values
X_train,X_test,Y_train,Y_test = train_test_split(X,y,test_size=0.2,random_state=0)


def get_input():
    studytime=st.number_input("Study Time")
    freetime=st.number_input('Free Time')
    health=st.number_input('Health')
    G1=st.number_input('First Marks')
    G2=st.number_input("Second Marks")
    absences= st.number_input("Absences")

    user_data={'studytime': studytime,
    'freetime':freetime,
    'health':health,
    'G1':G1, 
     'G2':G2,
     'absences':absences
     }
    

    features= pd.DataFrame(user_data,index=[0])
    return features
user_input =get_input()
st.subheader('User Input: ')
st.write(user_input)

model= LinearRegression()
model=model.fit(X_train, Y_train)

st.subheader("Model R2 Score on test data: ")
st.write(str(r2_score(Y_test,model.predict(X_test))))

st.subheader("Model Mean Square Error on test data: ")
st.write(str(mean_squared_error(Y_test,model.predict(X_test))))

predicted = model.predict(user_input)
st.write("Predicted Marks are: ",predicted)

    