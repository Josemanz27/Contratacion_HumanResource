from flask import Flask,request,render_template
import numpy as np
import pandas
import sklearn
import pickle

# Importar los modelos
model = pickle.load(open('model.pkl','rb'))
mm = pickle.load(open('minmaxscaler.pkl','rb'))

# crear flask
app = Flask(__name__)

@app.route('/')
def index():
    return render_template("index.html")

@app.route("/predict",methods=['POST'])
def predict():
    gender = request.form['gender']
    ssc_p = request.form['ssc_p']
    hsc_p = request.form['hsc_p']
    degree_p = request.form['degree_p']
    degree_p = request.form['degree_p']
    workex = request.form['workex']
    etest_p = request.form['etest_p']
    specialisation = request.form['specialisation']
    mba_p = request.form['mba_p']
    
    def transform(array):
        array[array == 'M'] = 1
        array[array == 'F'] = 0
        array[array == 'Yes'] = 1
        array[array == 'No'] = 0
        array[array == 'Placed'] = 1
        array[array == 'Not Placed'] = 0
        array[array == 'Mkt&HR'] = 1
        array[array == 'Mkt&Fin'] = 0

    def prediction(gender, ssc_p, hsc_p, degree_p, workex, etest_p, specialisation, mba_p):
        features = np.array([[gender, ssc_p, hsc_p, degree_p, workex, etest_p, specialisation, mba_p]])
        features_transformed = transform(features)
        features_transformed = mm.transform(features_transformed)
        prediction = model.predict(features_transformed)

        return prediction[0]
      
    single_pred = prediction(gender, ssc_p, hsc_p, degree_p, workex, etest_p, specialisation, mba_p)
    
    result =(f"La prediccion es: {single_pred}")
    return render_template('index.html',result = result)

# python main
if __name__ == "__main__":
    app.run(debug=True)