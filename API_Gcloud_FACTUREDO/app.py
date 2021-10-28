import numpy as np
import pandas as pd
from flask import Flask, request, jsonify, render_template, url_for
import pickle
import joblib
from sklearn.ensemble import RandomForestClassifier
import streamlit as st


# Path del modelo preentrenado
MODEL_PATH = 'models/pickle_model.pkl'


# Se recibe la imagen y el modelo, devuelve la predicción
def model_prediction(x_in, model):
    x = pd.DataFrame(np.asarray(x_in).reshape(1,-1))
#x = np.asarray(x_in).reshape(1,-1)
    preds=model.predict(x)
    ##
    if preds == 0:
        pred = 'Pagador'
    else:
        pred = 'No Pagador'
    return pred    


def main():
    model=''
    # Se carga el modelo
    if model=='':
        joblib.load(MODEL_PATH)
        #with open(MODEL_PATH, 'rb') as file:
           # model = pickle.load(file)
    
    # Título
    html_temp = """
    <h1 style="color:#181082;text-align:center;">SISTEMA DE PREDICCION DE PERFIL DE CLIENTE </h1>
    </div>
    """
    st.markdown(html_temp,unsafe_allow_html=True)

    # Lecctura de datos
    #Datos = st.text_input("Ingrese los valores de los features:")
    v1 = st.text_input("v1:")
    v3 = st.text_input("v3:")
    v23= st.text_input("v23:")
    v33= st.text_input("v33:")
    v10= st.text_input("v10:")
    v2 = st.text_input("v2:")
    v5 = st.text_input("v5:")
    v26= st.text_input("v26:")
    v9 = st.text_input("v9:")
    v4 = st.text_input("v4:")   
    
    # El botón predicción se usa para iniciar el procesamiento
    if st.button("Predicción :"): 
        #x_in = list(np.float_((Datos.title().split('\t'))))
        x_in =[np.float_(v1.title()),
                    np.float_(v3.title()),
                    np.float_(v23.title()),
                    np.float_(v33.title()),
                    np.float_(v10.title()),
                    np.float_(v2.title()),
                    np.float_(v5.title()),
                    np.float_(v26.title()), 
                    np.float_(v9.title()),                     
                    np.float_(v4.title())]
        predictS = model_prediction(x_in, model)
        st.success('EL CLIENTE TIENE PERFIL DE: {}'.format(predictS).upper())

if __name__ == '__main__':
    main()
