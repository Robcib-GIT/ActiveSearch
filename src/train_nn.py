'''
    Entrena una red neuronal a partir de una base de datos como la creada con create_db.py
    Jorge F. García-Samartín
    www.gsamartin.es
    31-03-2022
'''

import os
import pandas as pd
import tensorflow as tf
from datetime import datetime
import numpy as np

from sklearn.model_selection import train_test_split

from train_rf import estudiarErrores


'''-----------------------------------------------------------------------------------------------
    Funciones auxiliares
-----------------------------------------------------------------------------------------------'''

'''-----------------------------------------------------------------------------------------------
    Main
-----------------------------------------------------------------------------------------------'''
def main():
    # Directorio de trabajo
    direccion = os.path.dirname(os.path.abspath(__file__)) + '/'

    # Leer los datos
    database = '2022-03-29-21-55-35-database.csv'
    data = pd.read_csv(direccion + 'runs/detect/' + database, header=None)

    # División de los datos en train y test
    x_train, x_test, y_train, y_test = train_test_split (
                                            data.drop(columns = 17),
                                            data[17]
                                        )

    # Entrenar el modelo
    modelo = tf.keras.models.Sequential([
        tf.keras.layers.Dense(units = 100, activation = 'relu', input_shape = (x_train.shape[1],)),
        tf.keras.layers.Dense(units = 100, activation = 'relu'),
        tf.keras.layers.Dropout (0.2),
        tf.keras.layers.Dense(units = 1)
    ])
    modelo.compile(optimizer = 'adam', loss = 'mse', metrics = ['accuracy', 'mae', 'mse'])

    modelo.fit(x_train, y_train)

    # Predicciones
    predicciones = modelo.predict(x_test, batch_size = len(x_test.values))
    predicciones2 = pd.DataFrame(data = predicciones, index = x_test.index)
    predicciones2.columns = ['y_pred']

    # Damos nombre a y_test por elegancia
    y_test = y_test.to_frame()
    y_test.columns = ['y_test']

    # Estudiamos los errores
    rmse1, mae1, xyp_test1 = estudiarErrores(x_test, y_test, predicciones2, direccion = direccion, technique = 'nn')

    # Estudiamos los errores quitando las filas vacías
    print('Limpiamos los datos')
    rmse2, mae2, xyp_test2 = estudiarErrores(x_test, y_test, predicciones2, clean = True, direccion = direccion, technique = 'nn')

if __name__ == "__main__":
    main()