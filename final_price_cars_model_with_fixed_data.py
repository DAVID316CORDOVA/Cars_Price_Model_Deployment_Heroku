import pandas as pd 
import joblib

pd.options.display.float_format="{:.2f}".format

df=pd.read_csv("new_dataset_cars.csv")

#Defining X and y

X=df.drop(columns=["price"])
y=df[["price"]]


#Standardization the data

from sklearn.preprocessing import StandardScaler

scaler_x=StandardScaler()
scaler_y=StandardScaler()

scaler_x.fit(X)
scaler_y.fit(y)

X_scaled=pd.DataFrame(scaler_x.transform(X),columns=X.columns)
y_scaled=pd.DataFrame(scaler_y.transform(y),columns=y.columns)


#SOLVING THE PROBLEM WITH ARTIFICIAL NEURAL NETWORKS

#Importing the libraries
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

#Creating the model
model=Sequential()
model.add(Dense(256,input_dim=X.shape[1],activation="relu"))
model.add(Dense(128,activation="relu"))
model.add(Dense(64,activation="relu"))
model.add(Dense(32,activation="relu"))
model.add(Dense(16,activation="relu"))
model.add(Dense(8,activation="relu"))
model.add(Dense(1,activation="linear"))

#Compiling the model
model.compile(optimizer="adam", loss="mse")

#Fitting the model
history=model.fit(X_scaled,y_scaled,epochs=100,batch_size=32)

model.save("cars_price_final_model.h5")
joblib.dump(scaler_x,"scaler_x.pkl")
joblib.dump(scaler_y,"scaler_y.pkl")

#Saving a dictionary with all the columns

diccionario=dict(zip(X.columns,range(X.shape[1])))

joblib.dump(diccionario,open("indice_diccionario","wb"))

print("Diccionario de columnas:")

print(diccionario)



