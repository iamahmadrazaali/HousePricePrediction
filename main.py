from flask import Flask,render_template,request,redirect
import pandas as pd
from flask_cors import CORS,cross_origin
import pickle
import numpy as np
from num2words import num2words

app = Flask(__name__)
model=pickle.load(open('house_prediction_model.pkl','rb'))
ism = pd.read_csv("Cleaned_data.csv")

@app.route('/')
def index():
    locations = sorted(ism['location'].unique())
    # baths
    # bedrooms
    # Total_Area

    return render_template('index.html',locations=locations)

@app.route('/predict',methods=['POST'])
@cross_origin()
def predict():

    location=request.form.get('location')

    baths=request.form.get('baths')
    bedrooms=request.form.get('bedrooms')
    area=request.form.get('area')


    prediction = model.predict(pd.DataFrame(columns=['location','baths','bedrooms','Total_Area'],data=np.array([location,baths,bedrooms,area]).reshape(1,4)))
    # prediction = num2words(int(prediction), to = 'ordinal')
    print(prediction)

    # return prediction.capitalize()
    return str(int(prediction))

if __name__ == "__main__":
    app.run(debug=True)