import numpy as np
import pandas as pd
from flask import Flask, request, render_template, url_for
from keras.models import load_model
import joblib
diccionario=joblib.load(open("indice_diccionario.pkl","rb"))
app = Flask(__name__)
model = load_model("cars_price_final_model.h5")

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/predict',methods = ['POST'])
def predict():
    
    result=request.form
    year=result["year"]
    manufacturer=result["manufacturer"]
    condition=result["condition"]
    cylinders=result["cylinders"]
    fuel=result["fuel"]
    title_status=result["title_status"]
    transmission=result["transmission"]
    drive=result["drive"]
    size=result["size"]
    type=result["type"]
    paint_color=result["paint_color"]
    
    vector_zeros=np.zeros(len(diccionario))

    vector_zeros[0]=year

    vector_zeros[diccionario[str(manufacturer)]]=1

    vector_zeros[diccionario[str(condition)]]=1

    vector_zeros[diccionario[str(cylinders)]]=1

    vector_zeros[diccionario[str(fuel)]]=1

    vector_zeros[diccionario[str(title_status)]]=1

    vector_zeros[diccionario[str(transmission)]]=1

    vector_zeros[diccionario[str(drive)]]=1

    vector_zeros[diccionario[str(size)]]=1

    vector_zeros[diccionario[str(type)]]=1

    vector_zeros[diccionario[str(paint_color)]]=1

    
    escalador_x=joblib.load(open("scaler_x.pkl","rb"))
    escalador_y=joblib.load(open("scaler_y.pkl","rb"))

    data=pd.DataFrame(vector_zeros).T
    data.columns=[['year', 'acura', 'alfa-romeo', 'aston-martin', 'audi', 'bmw', 'buick',
       'cadillac', 'chevrolet', 'chrysler', 'datsun', 'dodge', 'ferrari',
       'fiat', 'ford', 'gmc', 'harley-davidson', 'hennessey', 'honda',
       'hyundai', 'infiniti', 'jaguar', 'jeep', 'kia', 'land rover', 'lexus',
       'lincoln', 'mazda', 'mercedes-benz', 'mercury', 'mini', 'mitsubishi',
       'morgan', 'nissan', 'pontiac', 'porche', 'ram', 'rover', 'saturn',
       'subaru', 'toyota', 'volkswagen', 'volvo', 'excellent', 'fair', 'good',
       'like new', 'new', 'salvage_x', '10 cylinders', '12 cylinders',
       '3 cylinders', '4 cylinders', '5 cylinders', '6 cylinders',
       '8 cylinders', 'diesel', 'electric', 'gas', 'hybrid', 'clean', 'lien',
       'missing', 'parts only', 'rebuilt', 'salvage_y', 'automatic', 'manual',
       '4wd', 'fwd', 'rwd', 'compact', 'full-size', 'mid-size', 'sub-compact',
       'SUV', 'bus', 'convertible', 'coupe', 'hatchback', 'mini-van',
       'offroad', 'pickup', 'sedan', 'truck', 'van', 'wagon', 'black', 'blue',
       'brown', 'custom', 'green', 'grey', 'orange', 'purple', 'red', 'silver',
       'white', 'yellow']]

    data_transformada=escalador_x.transform(data)
    prediction = model.predict(data_transformada,verbose=0)

    prediccion_real=escalador_y.inverse_transform(prediction)

    return render_template('home.html', prediction_text=f"Precio en US$ : {(prediccion_real[0][0]):.1f}")

if __name__ == '__main__':
    app.run(debug=True)
