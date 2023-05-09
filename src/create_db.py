'''
    Crea una base de datos con la que entrenar un random forest o una red neuronal.
    Coge imágenes y almacena las probabilidades de detección de un conjunto de objetos dado.
    Almacena también el tipo de habitación que es la imagen

    Jorge F. García-Samartín
    www.gsamartin.es
    24-03-2022
'''

import os
import numpy as np
from detect_jorge import (parse_opt, Detection)
from aux_functions import (findAll, getObjIds, getAllTags)
from datetime import datetime
from pycocotools.coco import COCO
from aux_functions import getAllTags

# Directorio de trabajo
dir = os.path.dirname(os.path.abspath(__file__)) + '/'
coco_annotation_file_path = dir + "../cocoapi/annotations/instances_train2017.json"
coco_annotation = COCO(annotation_file=coco_annotation_file_path)

# Configuramos las opciones del detector
opciones = parse_opt()
opciones.nosave = True
opciones.save_txt = False
opciones.save_conf = False
opciones.source = dir + '../cocoapi/images/train2017/'
#opciones.source = dir + 'data/images/tests'

# Seleccionamos las imágenes con las que vamos a trabajar
'''
query_id = coco_annotation.getCatIds(catNms=[objTags])[0]
img_ids = coco_annotation.getImgIds(catIds=[query_id])
'''

'''-----------------------------------------------------------------------------------------------
    Funciones auxiliares
-----------------------------------------------------------------------------------------------'''

'''-----------------------------------------------------------------------------------------------
    Identificación del tipo de habitación
-----------------------------------------------------------------------------------------------'''
# Parámetros de configuración
objTags = getAllTags()[2]
#numImgs = len(img_ids)                                      # Número de imágenes a procesar
numImgs = len(os.listdir(opciones.source))                  # Número de imágenes a procesar
numHabs = 5                                                 # Número de habitaciones
numObjs = len(objTags)                                      # Número de objetos
umbralHabitacion = 0.75                                     # Umbral para dar por buena la detección de una habitación
umbralHumano = 0.3                                          # Umbral de detección humano

# Creamos la base de datos
database = np.zeros((numImgs, numHabs + numObjs + 1))

opciones.weights = dir + 'weights/habitaciones.pt'
det = Detection(opt=opciones)

# Parámetros de configuración
#numImgs = len(img_ids)                                      # Número de imágenes a procesar
numImgs = len(os.listdir(opciones.source))                  # Número de imágenes a procesar
numHabs = 5                                                 # Número de habitaciones
numObjs = len(objTags)                                      # Número de objetos
umbralHabitacion = 0.75                                     # Umbral para dar por buena la detección de una habitación
umbralHumano = 0.3                                          # Umbral de detección humano

# Recorremos cada imagen de det.imgLabels y añadimos las conclusiones a la base de datos
# Para ello, miramos en cada imagen si hay detectado cocina, salón...y si es que sí, guardamos la probabilidad de que se haya detectado
for img in range(len(det.imgLabels)):
    # Recorremos los posibles tipos de cada habitación
    for tipo in range(0, numObjs):
        indxTipo = findAll(tipo, det.imgLabels[img])

        # Si hay al menos uno, guardamos su probabilidad
        if len(indxTipo):

            # Si hay más de uno, nos quedamos con el más alto
            if len(indxTipo) > 1:
                aux = np.sort(det.imgLabels[img], axis=0)
                indxTipo = findAll(tipo, aux)
                det.imgLabels[img][0] = aux[indxTipo[-1][0]]

            # Guardamos los resultados
            database[img][tipo] = det.imgLabels[img][indxTipo[-1][0]][1] if det.imgLabels[img][indxTipo[-1][0]][1] > umbralHabitacion else 0

'''-----------------------------------------------------------------------------------------------
    Detección de objetos
-----------------------------------------------------------------------------------------------'''
opciones.weights = dir + 'weights/yolov5s.pt'
det = Detection(opt=opciones)

# Hacemos lo mismo con objetos
objIds = getObjIds(objTags)
for img in range(len(det.imgLabels)):
    # Recorremos los posibles tipos de cada objeto
    for numTipo, tipo in enumerate(objIds):
        indxTipo = findAll(tipo, det.imgLabels[img])

        # Si hay al menos uno, guardamos su probabilidad
        if len(indxTipo):

            # Si hay más de uno, nos quedamos con el más alto
            if len(indxTipo) > 1:
                aux = np.sort(det.imgLabels[img], axis=0)
                indxTipo = findAll(tipo, aux)
                det.imgLabels[img][0] = aux[indxTipo[-1][0]]

            # Guardamos los resultados
            database[img][numHabs + numTipo] = det.imgLabels[img][indxTipo[-1][0]][1]

    # Si hay humano, lo anotamos como 1
    if findAll(0, det.imgLabels[img]):
        database[img][-1] = 1

# Guardamos database en un fichero CSV
np.savetxt(dir + 'results/' + datetime.now().strftime('%Y-%m-%d-%H-%M-%S') + '-database.csv', database, delimiter=',', fmt='%1.5f')