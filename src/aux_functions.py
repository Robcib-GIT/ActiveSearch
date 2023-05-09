'''
    Funciones auxiliares para la creación de la base de datos y el entrenamiento de RF
    Jorge F. García-Samartín
    www.gsamartin.es
    05-04-2022
'''

import yaml
import os
from pycocotools.coco import COCO
import status

# Directorio de trabajo
dir = os.path.dirname(os.path.abspath(__file__)) + '/'
if not(status.TEST or status.TAGS):
    coco_annotation_file_path = dir + "../cocoapi/annotations/instances_train2017.json"
    coco_annotation = COCO(annotation_file=coco_annotation_file_path)

# Devuelve la lista de etiquetas de habitaciones y objetos
def getAllTags ():
    with open( dir + 'data/HouseRooms.yaml', 'r') as stream:
        try:
            roomsInfo = yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            print(exc)
            
    roomTags = roomsInfo['names']
    objTags = ['person', 'bench', 'snowboard', 'skateboard', 'surfboard', 'chair', 'couch', 'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'refrigerator']
    allTags = roomTags + objTags

    return allTags, roomTags, objTags

# Encuentra todas las repeticiones de un elemento en una matriz
def findAll(element, matrix):
    result = []
    for i in range(len(matrix)):
        for j in range(len(matrix[i])):
            if matrix[i][j] == element:
                result.append([i, j])
    return result

# Devuelve los ids de los objetos con los que se va a trabajar (para poder usarlos en otro script)
def getObjIds(objTags):
    if not(status.TEST or status.TAGS):
        objIds = []
        for i in range(len(objTags)):
            objIds.append(coco_annotation.getCatIds(catNms=[objTags[i]])[0])
    else:
        objIds = [0, 15, 36, 41, 42, 62, 63, 64, 65, 67, 70, 72, 82]
    return objIds

# COCO get objects tags
def getObjTags(objIds):
    if not(status.TEST or status.TAGS):
        objTags = []
        for i in range(len(objIds)):
            objTags.append(coco_annotation.loadCats([objIds[i]])[0]["name"])
    else:
        objTags = ['person', 'bench', 'snowboard', 'skateboard', 'surfboard', 'chair', 'couch', 'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'refrigerator']
    return objTags