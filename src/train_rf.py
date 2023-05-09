'''
    Entrena un random forest a partir de una base de datos como la creada con create_db.py
    Jorge F. García-Samartín
    www.gsamartin.es
    31-03-2022
'''

import os
import pandas as pd
from datetime import datetime

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import RandomizedSearchCV

from aux_functions import getAllTags


'''-----------------------------------------------------------------------------------------------
    Funciones auxiliares
-----------------------------------------------------------------------------------------------'''
# Calcula el MAE
def calculaMAE (y_test, predicciones):
    errores = abs(predicciones.squeeze() - y_test)
    mae = 100 * sum(errores) / len(errores)
    return mae

# Calcula error cuadrático y MAE y genera CSV con los resultados
def estudiarErrores (x_test, y_test, predicciones, clean = False, direccion = '.', technique = 'rf'):  

    # Creamos los dataframe de entradas, realidad y predicciones
    if clean == True:
        aux = x_test.loc[~(x_test == 0).all(axis=1)]
        sufijo = '_clean'
    else:
        aux = x_test
        sufijo = ''

    xy_test = aux.join(y_test)
    xyp_test = xy_test.join(predicciones, rsuffix='_pred')

    # Redefinimos y_test y las predicciones para que sean coherentes si ha habido limpieza de datos
    y_test = xyp_test['y_test']
    predicciones = xyp_test['y_pred']

    # Cálculo del error cuadrático
    rmse = mean_squared_error(
            y_true  = y_test,
            y_pred  = predicciones,
            squared = False
        )
    print(f"El error (rmse) de test es: {rmse}")

    # Cálculo del MAE
    mae = calculaMAE(y_test, predicciones)
    precision = 100 - mae
    print(f"La precisión es: {precision} %")

    # Guardamos los datos en un CSV
    xyp_test.to_csv(direccion + '/results/' + datetime.now().strftime('%Y-%m-%d-%H-%M-%S') + '-xyp_test' + technique + sufijo + '.csv')

    # Devolvemos los errores y la tabla de entradas y salidas
    return rmse, mae, xyp_test

# Representa la importancia de las distintas variables del RF
def representarImportancia (rf, feature_list):
    importances = list(rf.feature_importances_)
    # List of tuples with variable and importance
    feature_importances = [(feature, round(importance, 2)) for feature, importance in zip(feature_list, importances)]
    # Sort the feature importances by most important first
    feature_importances = sorted(feature_importances, key = lambda x: x[1], reverse = True)
    # Print out the feature and importances 
    [print('Variable: {:20} Importance: {}'.format(*pair)) for pair in feature_importances]

# Dibuja el árbol de decisión
def plotTree(rf, feature_list, direccion, train_features, train_labels):
    # Import tools needed for visualization
    from sklearn.tree import export_graphviz
    import pydot
    # Pull out one tree from the forest
    tree = rf.estimators_[5]
    # Import tools needed for visualization
    from sklearn.tree import export_graphviz
    import pydot
    # Pull out one tree from the forest
    tree = rf.estimators_[5]
    # Export the image to a dot file
    export_graphviz(tree, out_file = 'tree.dot', feature_names = feature_list, rounded = True, precision = 1)
    # Use dot file to create a graph
    (graph, ) = pydot.graph_from_dot_file('tree.dot')
    # Write graph to a png file
    graph.write_png(direccion + 'results/' + datetime.now().strftime('%Y-%m-%d-%H-%M-%S') + 'tree.png')
    # Limit depth of tree to 3 levels
    rf_small = RandomForestRegressor(n_estimators=10, max_depth = 3)
    rf_small.fit(train_features, train_labels)
    # Extract the small tree
    tree_small = rf_small.estimators_[5]
    # Save the tree as a png image
    export_graphviz(tree_small, out_file = 'small_tree.dot', feature_names = feature_list, rounded = True, precision = 1)
    (graph, ) = pydot.graph_from_dot_file('small_tree.dot')
    graph.write_png(direccion + 'results/' + datetime.now().strftime('%Y-%m-%d-%H-%M-%S') + 'small_tree.png');

# Entrena el árbol de decisión
def entrenarArbol (x_train, y_train):    

    # Entrenar el modelo
    modelo = RandomForestRegressor(
                n_estimators = 10,
                criterion    = 'squared_error',
                max_depth    = 15,
                max_features = 'auto',
                oob_score    = False,
                n_jobs       = -1
            )
    '''modelo = RandomForestRegressor(
                n_estimators = 90,
                criterion    = 'squared_error',
                max_depth    = 11,
                max_features = 'sqrt',
                oob_score    = False,
                n_jobs       = -1
            )
    modelo = RandomForestRegressor(
                n_estimators = 10,
                criterion    = 'squared_error',
                max_depth    = 44,
                max_features = 'auto',
                oob_score    = False,
                n_jobs       = -1
            )'''


    modelo.fit(x_train, y_train)

    return modelo

'''-----------------------------------------------------------------------------------------------
    Main
-----------------------------------------------------------------------------------------------'''
def main():
    # Directorio de trabajo
    direccion = os.path.dirname(os.path.abspath(__file__)) + '/'

    # Configuración
    ajustar = False

    # Leer los datos
    database = '2022-04-01-02-33-34-database.csv'
    data = pd.read_csv(direccion + 'results/' + database, header=None)
    tags = getAllTags()[0]
    tags.remove('person')
    data.columns = tags + ['y_test']

    # División de los datos en train y test
    x_train, x_test, y_train, y_test = train_test_split (
                                            data.drop(columns = 'y_test'),
                                            data['y_test']
                                        )
    
    # Entrenamos el modelo
    modelo = entrenarArbol(x_train, y_train)

    # Predicciones
    predicciones = modelo.predict(X = x_test)
    predicciones2 = pd.DataFrame(data = {'y_pred': predicciones}, index = x_test.index)
    predicciones2.columns = ['y_pred']

    # Damos nombre a y_test por elegancia
    y_test = y_test.to_frame()
    y_test.columns = ['y_test']

    # Estudiamos los errores
    rmse1, mae1, xyp_test1 = estudiarErrores(x_test, y_test, predicciones2, direccion = direccion)

    # Estudiamos los errores quitando las filas vacías
    print('Limpiamos los datos')
    rmse2, mae2, xyp_test2 = estudiarErrores(x_test, y_test, predicciones2, clean = True, direccion = direccion)

    # Representamos la importancia de las distintas variables
    representarImportancia(modelo, list(data.drop(columns = 'y_test').columns))

    # Dibujamos el árbol
    plotTree(modelo, list(data.drop(columns = 'y_test').columns), direccion, x_train, y_train)

    # Lista de todos los parámetros
    # print(modelo.get_params().keys())

    # Ajuste de hiperparámetros
    if ajustar:
        parametros = {
            'max_depth': range(10,50),
            'max_features': ['auto', 'sqrt', 'log2'],
            'n_estimators': range(10,100,10)
        }

        clf = RandomizedSearchCV(modelo, parametros, scoring = 'neg_mean_absolute_error', cv = 5, n_iter = 100, verbose = 3, error_score = 'raise')
        clf.fit(x_train, y_train)
        print(clf.best_estimator_ )
        print(clf.best_params_)
        print(clf.best_score_)

        print('fin')

if __name__ == "__main__":
    main()