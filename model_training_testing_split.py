import pandas as pd

pd.options.display.float_format="{:.2f}".format

cars=pd.read_csv("cars.csv")

index_names=cars[(cars["manufacturer"]=="other") | (cars["condition"]=="other") | (cars["cylinders"]=="other") | (cars["fuel"]=="other") | (cars["title_status"]=="other") | (cars["transmission"]=="other") | (cars["drive"]=="other") | (cars["size"]=="other") | (cars["type"]=="other") | (cars["paint_color"]=="other") ].index

df=cars.drop(index_names)

#Finding the categorical features

categorical=df.select_dtypes(include="object").columns

#Filling the null values with the mode of each feature

for line in categorical:
    df[line]=df[line].fillna(df[categorical].mode()[line][0])  
    
#Creating new features with the help of get_dummies

for column in categorical:
    nuevas_features = pd.get_dummies(df[column])
    df= pd.merge(
        left=df,
        right=nuevas_features,
        left_index=True,
        right_index=True,
    )
    df = df.drop(columns=column)

print(df.head())

#The dataframe will work with prices between 5000 and 40000 and a year greater than 1945

df=df[(df["price"].between(5000,40000,inclusive="both")) & (df["year"]> 1945)]

df.to_csv("new_dataset_cars.csv",index=False)

#Defining X and y

X=df.drop(columns=["price"])
y=df[["price"]]

#Splitting the data in train and test
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.3, random_state=42)

#Standardization the data

from sklearn.preprocessing import StandardScaler

scaler_x=StandardScaler()
scaler_y=StandardScaler()

scaler_x.fit(X_train)
scaler_y.fit(y_train)

X_train=pd.DataFrame(scaler_x.transform(X_train),columns=X_train.columns)
X_test=pd.DataFrame(scaler_x.transform(X_test),columns=X_test.columns)

y_train=pd.DataFrame(scaler_y.transform(y_train),columns=y_train.columns)
y_test=pd.DataFrame(scaler_y.transform(y_test),columns=y_test.columns)

#SOLVING THE PROBLEM WITH ARTIFICIAL NEURAL NETWORKS

#Importing the libraries
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.callbacks import EarlyStopping

early=EarlyStopping(monitor="val_loss", patience=10)

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
history=model.fit(X_train,y_train,epochs=100,batch_size=32,validation_split=0.25, callbacks=[early])


print("Model Evaluation - MSE :",(model.evaluate(X_test,y_test,verbose=0)))
      
#Evaluating the neural network

from sklearn.metrics import mean_absolute_error,mean_squared_error,r2_score

y_pred=pd.DataFrame(scaler_y.inverse_transform(model.predict(X_test,verbose=0).reshape(-1,1)),columns=y_test.columns)

original_y_test=pd.DataFrame(scaler_y.inverse_transform(y_test).reshape(-1,1),columns=y_test.columns)

mae=mean_absolute_error(original_y_test,y_pred)

mse=mean_squared_error(original_y_test,y_pred)

r2=r2_score(original_y_test,y_pred)

print("MAE : %f , MSE : %f , R2 : %f" % (mae,mse,r2))
