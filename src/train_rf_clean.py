'''
    Entrena un random forest a partir de una base de datos como la creada con create_db.py
    Los datos ya están limpios 

    Jorge F. García-Samartín
    www.gsamartin.es
    31-03-2022
'''

import os
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import RandomizedSearchCV

from train_rf import estudiarErrores
from aux_functions import getAllTags

'''-----------------------------------------------------------------------------------------------
    Funciones auxiliares
-----------------------------------------------------------------------------------------------'''

'''-----------------------------------------------------------------------------------------------
    Main
-----------------------------------------------------------------------------------------------'''
# Directorio de trabajo
direccion = os.path.dirname(os.path.abspath(__file__)) + '/'

# Leer los datos
database = '2022-03-29-21-55-35-database.csv'
data = pd.read_csv(direccion + 'runs/detect/' + database, header=None)
tags = getAllTags()[0]
data.columns = tags + ['y_test']

# Quitamos las columnas que hemos visto que son menos importantes
data = data.drop(columns = ['skateboard', 'toilet', 'refrigerator', 'tv', 'surfboard'])

# División de los datos en train y test
x_train, x_test, y_train, y_test = train_test_split (
                                        data.drop(columns = 'y_test'),
                                        data['y_test']
                                    )

# Limpiamos los datos
x_test = x_test.loc[~(x_test == 0).all(axis=1)]
xy_test = x_test.join(pd.DataFrame({'y_test': y_test}))

# Redefinimos y_test para que sea coherente con la limpieza de datos
y_test = xy_test['y_test']

# Entrenar el modelo
modelo = RandomForestRegressor(
            n_estimators = 10,
            criterion    = 'squared_error',
            max_depth    = None,
            max_features = 'auto',
            oob_score    = False,
            n_jobs       = -1
         )
'''modelo = RandomForestRegressor(
            n_estimators = 30,
            criterion    = 'squared_error',
            max_depth    = 11,
            max_features = 'log2',
            oob_score    = False,
            n_jobs       = -1
         )'''

modelo.fit(x_train, y_train)

# Predicciones
predicciones = modelo.predict(X = x_test)
predicciones2 = pd.DataFrame(data = {'y_pred': predicciones}, index = x_test.index)

# Estudiamos los errores
rmse1, mae1, xyp_test1 = estudiarErrores(x_test, y_test, predicciones2, direccion = direccion)

'''# Prueba con una mesa
prueba = np.zeros(shape = (1, len(data.columns) - 1))
prueba[0][8] = 0.97
predPrueba = modelo.predict(X = prueba)
print(predPrueba)'''

'''
# Ajuste de hiperparámetros
parametros = {
    'max_depth': range(10,50),
    'max_features': ['auto', 'sqrt', 'log2'],
    'n_estimators': range(10,100,10)
}

clf = RandomizedSearchCV(modelo, parametros,cv = 5, n_iter = 100, verbose = 3, error_score = 'raise')
clf.fit(x_train, y_train)
print(clf.best_estimator_ )
print(clf.best_params_)
print(clf.best_score_)
'''
