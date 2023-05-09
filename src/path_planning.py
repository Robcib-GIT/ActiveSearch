#!/usr/bin/env python
'''
    Ejecuta el algoritmo de path planning para la deteccion de personas
    Jorge F. García-Samartín
    www.gsamartin.es
    19-04-2022
'''

import os
import math
import pandas as pd
import numpy as np
import sklearn as sk
import rospy
import random

from sensor_msgs.msg import CompressedImage
from nav_msgs.msg import Odometry
from nav_msgs.msg import OccupancyGrid
from visualization_msgs.msg import Marker

import status
from train_rf import entrenarArbol
from detect_jorge import Detection
from aux_functions import (findAll, getObjIds, getAllTags)
from robotControl import Robot

# Clase con las opciones del detector
class DetOptions:
    def __init__(self, direction):
        self.source = 'ROS'
        self.weights = direction + 'weights/yolov5s.pt'
        self.data = direction + 'data/coco128.yaml'
        self.imgsz = [640, 640]
        self.conf_thres = 0.3
        self.iou_thres = 0.45
        self.max_det = 1000
        self.device = 'cpu'
        self.view_img = True
        self.save_crop = True
        self.nosave = False
        self.save_txt = True
        self.save_conf = True
        self.save_array = True
        self.classes = 0
        self.agnostic_nms = True
        self.augment = True
        self.visualize = True
        self.update = True
        self.project = direction + 'runs/detect'
        self.name = 'exp'
        self.exist_ok = True
        self.line_thickness = 3
        self.hide_labels = False
        self.hide_conf = False
        self.half = True
        self.dnn = True
        self.ros = True
        self.camera = False


# Clase planificador
class Planner:

    '''-----------------------------------------------------------------------------------------------
        Constructor de la clase
    -----------------------------------------------------------------------------------------------'''
    def __init__ (self):
        # Parámetros de configuracion
        self.params = {
            'dir' : os.path.dirname(os.path.abspath(__file__)) + '/',         # Directorio de trabajo
            'database' : '2022-04-01-02-33-34-database.csv',                  # Base de datos para entrenar
            'nPoints' : 3,                                                    # Número de puntos aleatorios a considerar en cada iteración
            'dist' : 5.0,                                                     # Distancia a los puntos aleatorios
            'candRadius' : 2,                                                 # Distancia alrededor de la cual explorar los candidatos
            'personID' : 0,                                                   # ID que YOLO asocia a las personas detectadas
            'doorTol' : math.pi/3,                                            # Ángulo (en radianes) que girar si no se detecta puerta en esa posición
            'nRooms' : 1,                                                     # Número de habitaciones de la casa
            'nPossDoors' : 15,                                                # Número de puntos a los que dirigirse antes de considerar que se está rescatando en una única habitación
        }

        self.doors = []                                                     # Puertas de las habitaciones
        self.humans = []                                                    # Posiciones de los humanos
        self.candidates = []                                                # Puntos candidatos a ser explorados
        self.detections = []                                                # Resultados de aplicar detect_jorge a cada imagen
        
        self.allTags, self.roomTags, self.objTags = getAllTags()            # Lista de habitaciones y objetos
        self.numHabs = len(self.roomTags)                                   # Número de habitaciones
        self.objIds = getObjIds(self.objTags)                               # IDs de los objetos con los que se trabaja

        # Creación del detector
        self.opciones = DetOptions(self.params['dir'])                      # Opciones del detector
        self.detector = Detection(opt=self.opciones)

        # Vemos qué topic de ROS usar
        if self.detector.opt.camera == True:
            self.imgTopic = "/cv_camera/image_raw/compressed"
        else:
            self.imgTopic = "/camera/color/image_raw/compressed" # En Gazebo, poner /realsense/color/image_raw/compressed"

        # Detector de puertas
        self.opcDoors = DetOptions(self.params['dir'])
        self.opcDoors.data = self.params['dir'] + 'data/doors.yaml'
        self.opcDoors.weights = self.params['dir'] + 'weights/doors.pt'
        self.opcDoors.conf_thres = 0.2
        self.detDoors = Detection(opt=self.opcDoors)

        # Detector de habitaciones
        self.opcRooms = DetOptions(self.params['dir'])
        self.opcRooms.data = self.params['dir'] + 'data/HouseRooms.yaml'
        self.opcRooms.weights = self.params['dir'] + 'weights/habitaciones.pt'
        self.detRooms = Detection(opt=self.opcRooms)

        # Publicador de markers para RViz
        self.marker_pub = rospy.Publisher('/visualization_marker', Marker, queue_size=10)
        self.markerID = 0           # Para evitar markers con IDs repetidos

        # Robot
        self.robot = Robot(dir = self.params['dir'])

    '''-----------------------------------------------------------------------------------------------
        Callback de suscripción al mapa
    -----------------------------------------------------------------------------------------------'''
    def readMap(self):
        # Convertir el mapa (tupla) en una lista
        self.map = []
        for x in range(0, self.subMap.info.height):
            self.map.append(list(self.subMap.data[x*self.subMap.info.width:(x+1)*self.subMap.info.width]))

        # Obtener la posición de la base
        self.base = self.subMap.info.origin.position
        if self.subMap.info.origin.position.z != 0:
            print('El mapa está rotado')

    '''-----------------------------------------------------------------------------------------------
        Callback de suscripción a la imagen
    -----------------------------------------------------------------------------------------------'''
    def readImg(self):
        self.detector.img = self.subImage.data
        self.detector.makeInference()

    '''-----------------------------------------------------------------------------------------------
        Publicación de un marcador en el mapa de RVIZ
    -----------------------------------------------------------------------------------------------'''
    def publishPoint(self, pos, markerType, time=60):

        # Creamos el marcador
        marker = Marker()

        # Asociamos cada tipo a un color
        if markerType == 'Point':
            r, g, b = (0, 1, 0)
            figure = marker.SPHERE
        elif markerType == 'Human':
            r, g, b = (0, 0, 1)
            figure = marker.ARROW
        elif markerType == 'Door':
            r, g, b = (1, 0, 0)
            figure = marker.CUBE
        else:
            r, g, b = (1, 1, 0)
            figure = marker.CYLINDER

        # Configuración general del marcador
        marker.header.frame_id = 'map'
        marker.id = self.markerID
        marker.type = figure
        marker.action = marker.ADD
        marker.pose.position.x = pos[0]
        marker.pose.position.y = pos[1]
        marker.pose.position.z = 0
        marker.pose.orientation.x = 0.0
        marker.pose.orientation.y = 0.0
        marker.pose.orientation.z = 0.0
        marker.pose.orientation.w = self.robot.odom.pose.pose.orientation.w
        marker.scale.x = 0.2
        marker.scale.y = 0.2
        marker.scale.z = 0.2
        marker.color.a = 1.0 # Canal alpha
        marker.color.r = r
        marker.color.g = g
        marker.color.b = b
        marker.lifetime = rospy.Duration(time)

        # Excepción para los humanos: no ponemos un punto, sino que apuntamos hacia ellos
        if markerType == 'Human':
            
            # Orientación
            w, x, y, z = self.robot.euler2quat([0, 0, self.robot.w])
            marker.pose.orientation.x = x
            marker.pose.orientation.y = y
            marker.pose.orientation.z = z
            marker.pose.orientation.w = w

            # Tamaño
            marker.scale.x = 1
            marker.scale.y = 0.2
            marker.scale.z = 0.2

        # Publicamos el marcador y actualizamos el ID para el siguiente
        self.marker_pub.publish(marker)
        self.markerID += 1
    
    '''-----------------------------------------------------------------------------------------------
        Chequea si un punto del mapa está explorado
    -----------------------------------------------------------------------------------------------'''
    def isExplored(self, i, j):
        return not(self.map[j][i] == -1)

    '''-----------------------------------------------------------------------------------------------
        Chequea si un punto del mapa está libre u ocupado
    -----------------------------------------------------------------------------------------------'''
    def isFree(self, i, j):
        return self.map[j][i] == 0

    '''-----------------------------------------------------------------------------------------------
        Hace que el robot no se salga de los límites de la nave
    -----------------------------------------------------------------------------------------------'''
    def adjustNave(self, x, y):
        if x > 6:
            x = 6
        elif x < -0.45:
            x = 0.45

        if y > 2.8:
            y = 2.8
        elif y < -4.8:
            y = 4.8

        return x, y

    '''-----------------------------------------------------------------------------------------------
        Convierte de coordenadas en metros a índices de matriz de mapa
    -----------------------------------------------------------------------------------------------'''
    def coordToIndex(self, x, y):
        temp = [int((x-self.base.x) / self.subMap.info.resolution), int((y-self.base.y) / self.subMap.info.resolution)]
        # Hacemos que temp esté comprendido entre 0 y subMap.info.width
        if temp[0] < 0:
            temp[0] = 0
        if temp[0] >= self.subMap.info.width:
            temp[0] = self.subMap.info.width - 1
        if temp[1] < 0:
            temp[1] = 0
        if temp[1] >= self.subMap.info.height:
            temp[1] = self.subMap.info.height - 1
        return temp[0], temp[1]

    '''-----------------------------------------------------------------------------------------------
        Convierte de índices de matriz de mapa a coordenadas en metros
    -----------------------------------------------------------------------------------------------'''
    def indexToCoord(self, i, j):
        return (i*self.subMap.info.resolution) + self.base.x, (j*self.subMap.info.resolution) + self.base.y

    '''-----------------------------------------------------------------------------------------------
        Entrenar el modelo
    -----------------------------------------------------------------------------------------------'''
    def train (self):

        # Leer los datos
        self.data = pd.read_csv(self.params['dir'] + 'results/' + self.params['database'], header=None)
        tags = getAllTags()[0][1:]
        self.data.columns = tags + ['y_test']

        # División de los datos en train y test
        self.x_train, self.x_test, self.y_train, self.y_test = sk.model_selection.train_test_split (
                                            self.data.drop(columns = 'y_test'),
                                            self.data['y_test']
                                        )

        # Entrenamos el modelo
        self.modelo = entrenarArbol(self.x_train.values, self.y_train)

    '''-----------------------------------------------------------------------------------------------
        Recibe y gestiona los tópicos de recepción de puertas en el mapa
    -----------------------------------------------------------------------------------------------'''
    def detectDoors(self):
        self.detDoors.img = rospy.wait_for_message(self.imgTopic, CompressedImage).data
        self.detDoors.imgLabels = []
        self.detDoors.makeInference()
        self.convertDetectionDoors()

    '''-----------------------------------------------------------------------------------------------
        Localiza las puertas en el mapa
    -----------------------------------------------------------------------------------------------'''
    def convertDetectionDoors (self):

        # Parámetros de configuración
        stepSize = 0.1          # Distancia entre pasos al movernos por el mapa
        camAngle = math.pi/2    # Ángulo focal de la cámara (en radianes)

        
        for img in range(len(self.detDoors.imgLabels)):

            # Nos quedamos con las puertas de tipo 0 (door)
            indxTipo = findAll(0, self.detDoors.imgLabels[img])

            # Para cada puerta detectada
            for indxDoor in indxTipo:
                coords = self.detDoors.imgLabels[0][indxDoor[0]][2:4]

                # Ángulo (absoluto) en el que se encuentra la puerta
                angle = self.robot.w + camAngle * (coords[0] - 0.5)

                # Estimamos la posición en la que está la puerta buscando obstáculos en el mapa
                x = self.robot.x
                y = self.robot.y
                while self.isFree(*self.coordToIndex(x, y)):
                    x = x + stepSize * math.cos(angle)
                    y = y + stepSize * math.sin(angle)

                    if x == 0 or y == 0:
                        break
                    elif x >= self.subMap.info.width * self.subMap.info.resolution or y >= self.subMap.info.height * self.subMap.info.resolution:
                        break
                
                # Guardamos la puerta
                self.doors.append([x,y,0])

                # Publicamos la puerta en el mapa
                self.publishPoint([x,y], 'Door')


    '''-----------------------------------------------------------------------------------------------
        Convierte los datos de detección de habitaciones en un formato legible por el modelo
    -----------------------------------------------------------------------------------------------'''
    def convertDetectionRooms (self):

        for img in range(len(self.detRooms.imgLabels)):
            # Recorremos los posibles tipos de cada habitación
            for tipo in range(self.numHabs):
                indxTipo = findAll(tipo, self.detRooms.imgLabels[img])

                # Si hay al menos uno, guardamos su probabilidad
                if len(indxTipo):

                    # Si hay más de uno, nos quedamos con el más alto
                    if len(indxTipo) > 1:
                        aux = np.sort(self.detRooms.imgLabels[img], axis=0)
                        indxTipo = findAll(tipo, aux)
                        self.detRooms.imgLabels[img][0] = aux[indxTipo[-1][0]]

                    self.roomType[tipo] = self.detRooms.imgLabels[img][indxTipo[-1][0]][1]

    '''-----------------------------------------------------------------------------------------------
        Convierte los datos de detección de objetos en un formato legible por el modelo
    -----------------------------------------------------------------------------------------------'''
    def convertDetection (self):

        # Array con los humanos detectados en la exploración
        humans = []

        # Datos de la detección convertida
        lastDet = [0] * len(self.allTags)

        # Metemos los datos de las habitaciones
        lastDet[0:self.numHabs] = self.roomType

        for img in range(len(self.detector.imgLabels)):
            # Recorremos los posibles tipos de cada objeto
            for numTipo, tipo in enumerate(self.objIds[1:]):
                indxTipo = findAll(tipo, self.detector.imgLabels[img])

                # Si hay al menos uno, guardamos su probabilidad
                if len(indxTipo):

                    # Si hay más de uno, nos quedamos con el más alto
                    if len(indxTipo) > 1:
                        aux = np.sort(self.detector.imgLabels[img], axis=0)
                        indxTipo = findAll(tipo, aux)
                        self.detector.imgLabels[img][0] = aux[indxTipo[-1][0]]

                    lastDet[self.numHabs + numTipo] = self.detector.imgLabels[img][indxTipo[-1][0]][1]

        # Buscamos humanos
        numHumans = len(findAll(self.params['personID'], self.detector.imgLabels[img]))
        if numHumans:
            humans.append(self.robot.pos)
            print(str(numHumans) + ' humanos detectado desde la posición (' + str(self.robot.x) + ', ' + str(self.robot.y) + ')' + ' con orientación ' + str(self.robot.w))
            self.publishPoint([self.robot.x, self.robot.y], 'Human', 0)

        # Guardamos los resultados
        self.detections.append(lastDet)

        # Guardamos los humanos detectados en esta habitación
        self.humans.extend(humans)

    '''-----------------------------------------------------------------------------------------------
        Calcula el número de puntos sin explorar en un cuadrado de lado 2R en torno a un punto
    -----------------------------------------------------------------------------------------------'''
    def countNotExploredPoints(self, x, y, R):
        i, j = self.coordToIndex(x, y)
        numCells = int(round(R / self.subMap.info.resolution))
        width = np.array(range(i - numCells, i + numCells + 1))
        height = np.array(range(j - numCells, j + numCells + 1))

        # Ajustamos width y height a los límites del mapa
        width = width[(width > 0) & (width < self.subMap.info.width)]
        height = height[(height > 0) & (height < self.subMap.info.height)]
        area = len(width) * len(height)

        count = 0
        for l in width:
            for m in height:
                if not self.isExplored(l,m):
                    count += 1
        
        return count, count/area

    '''-----------------------------------------------------------------------------------------------
        Calcula el fitness de un candidato a siguiente punto de la exploración
    -----------------------------------------------------------------------------------------------'''
    def calculateFitness(self, candidate):
        # Coeficientes de la función de fitness
        alpha = [0.1, 2, 0.7]

        # Volumen potencial explorable
        vol = self.countNotExploredPoints(candidate[0], candidate[1], self.params['candRadius'])[1]

        # Orientamos el robot
        if status.TEST:
            print('Posición detectada: ' + str(self.robot.x) + ' , ' + str(self.robot.y))
            print('Posición y orientación del candidato: ' + str(candidate[0]) + ' , ' + str(candidate[1]) + ' , ' + str(candidate[2]))
        w0 = self.robot.w
        self.publishPoint([candidate[0], candidate[1]], 'Candidate', 10)
        self.robot.moveTo(self.robot.x, self.robot.y, candidate[2], onlyOr = True)
        if status.VERBOSE and status.TEST:
            print('Ángulo teórico: {:f}. Ángulo alcanzado: {:f}'.format(w0 + candidate[2], self.robot.w))

        # Ganancia de información (dada por el modelo entrenado)
        self.subImage = rospy.wait_for_message(self.imgTopic, CompressedImage)
        self.readImg()
        self.convertDetection()
        info = self.modelo.predict(X = [self.detections[-1][1:]])[0]

        # Distancia al punto
        pos = np.array([self.robot.x, self.robot.y])
        distance = np.linalg.norm(pos - np.array(candidate[0:2]))

        return  alpha[0]*vol + alpha[1]*info + alpha[2]*distance/self.params['dist']

    '''-----------------------------------------------------------------------------------------------
        Explorar
    -----------------------------------------------------------------------------------------------'''
    def exploreRoom (self):

        # Leemos el mapa
        self.subMap = rospy.wait_for_message('/map', OccupancyGrid)
        self.readMap()

        # Actualizamos la posición del robot
        self.odom = rospy.wait_for_message('/odom', Odometry)
        self.robot.readPos()
        if status.VERBOSE:
            print('El robot está en el punto: ({:f}, {:f}, {:f})'.format(self.robot.x, self.robot.y, self.robot.w))
        
        # Listado de candidatos recogidos en esta iteración
        provCandidates = []
        
        # Cogemos un número nPoints de puntos aleatorios a distancia dist
        remainPoints = self.params['nPoints']
        totalPoints = 0
        while remainPoints:
            # Generamos un punto aleatorio
            x = random.uniform(self.robot.x - self.params['dist'], self.robot.x + self.params['dist'])
            y = random.uniform(self.robot.y - self.params['dist'], self.robot.y + self.params['dist'])
            
            # Si el punto está en los límites y es válido, lo guardamos
            if status.NAVE:
                x, y = self.adjustNave(x, y)            
            if self.isFree(*self.coordToIndex(x, y)):
                angle = math.atan2(y - self.robot.y, x - self.robot.x)
                provCandidates.append([x,y,angle,0])
                remainPoints -= 1
            totalPoints += 1

        if status.VERBOSE:
            print('Número total de puntos: ' + str(totalPoints))
        
        # Ordenamos a los candidatos en función del parámetro angle
        provCandidates.sort(key=lambda x: self.robot.wrapToPi(x[2]), reverse=False)

        # Calculamos el fitness de los nuevos puntos
        for candidate in provCandidates:
            candidate[3] = self.calculateFitness(candidate)
            self.candidates.append(candidate)

        # Cogemos el punto de mayor fitness y movemos el robot allí
        self.robot.readPos()
        self.candidates.sort(key=lambda x: x[3], reverse=True)
        if status.VERBOSE:
            print('Movimiento al punto: ({:f}, {:f}, {:f})'.format(self.candidates[0][0], self.candidates[0][1], self.candidates[0][2]))
        self.publishPoint([self.candidates[0][0], self.candidates[0][1]], 'Point', 0)
        self.robot.moveTo(self.candidates[0][0], self.candidates[0][1], self.candidates[0][2])
        if status.VERBOSE:
            print('Ha llegado al punto: ({:f}, {:f}, {:f})'.format(self.robot.x, self.robot.y, self.robot.w))

        # Si el valor de la mejor celda es muy bajo (¿debajo del 0.54 que sale con todo 0?), irse por donde ha venido o a la otra puerta
        # Dar una vuelta para buscar puertas
    
    def explore(self):

        # Leemos el mapa
        self.subMap = rospy.wait_for_message('/map', OccupancyGrid)
        self.readMap()

        # Actualizamos la posición del robot
        self.odom = rospy.wait_for_message('/odom', Odometry)
        self.robot.readPos()

        # Borramos los markers anteriores
        marker = Marker()
        marker.action = marker.DELETEALL

        # Rescate
        notFound = 0
        numExplor = 0
        notFound += 1       # Número de veces en las que no se han encontrado peurtas
        while numExplor < self.params['nRooms']:

            # Búsqueda de puertas
            while not len(self.doors):

                # Modo de pruebas para funcionar en una sola habitación
                if not status.SEARCH_DOORS:
                    print('Modo de una habitación')
                    door = [self.robot.x, self.robot.y, 0]
                    numExplor += 1
                    break

                # Damos como mucho una vuelta alrededor de la posición actual
                numIntentos = math.ceil(2*math.pi / self.params['doorTol'])
                for intento in range(numIntentos):
                    self.robot.moveTo(self.robot.x, self.robot.y, self.robot.w + self.params['doorTol'], onlyOr = True)
                    self.detectDoors()

                # Si se han encontrado, nos quedamos con la más cercana
                numDoors = len(self.doors)
                if numDoors > self.params['nRooms']:
                    self.params['nRooms'] = numDoors

                if numDoors:
                    print('Movimiento a puerta')
                    for door in self.doors:
                        door[2] = math.sqrt((door[0]-self.robot.x)**2 + (door[1]-self.robot.y)**2)
                        self.publishPoint([door[0], door[1]], 'Door', 10)
                    self.doors.sort(key=lambda x: x[2], reverse=False)
                    door = self.doors.pop(0)
                    numExplor += 1
                    break

                # Si no se han encontrado puertas, buscamos más
                else:
                    print('Buscando nuevas puertas') 
                    # Generamos un punto aleatorio y nos movemos a él
                    x = random.uniform(self.robot.x - self.params['dist'], self.robot.x + self.params['dist'])
                    y = random.uniform(self.robot.y - self.params['dist'], self.robot.y + self.params['dist'])  
                    self.publishPoint([x, y], 'Point', 10)
                    self.robot.moveTo(x, y)

                    notFound += 1
                    if notFound == self.params['nPossDoors']:
                        door = [self.robot.x, self.robot.y, 0]
                        print('Sólo hay una habitación')
                        break

            # Movimiento a la puerta
            #self.robot.moveTo(door[0], door[1])
            input('Hay que atravesar la puerta. Cuando se haya atravesado, pulsa una tecla para continuar')

            # Guardar tipo de habitación y explorarla
            self.roomType = np.zeros(self.numHabs)
            self.detRooms.img = rospy.wait_for_message(self.imgTopic, CompressedImage).data
            self.detRooms.makeInference()
            self.convertDetectionRooms()
            for i in range(100):
                self.exploreRoom()
            break

'''-----------------------------------------------------------------------------------------------
    Main
-----------------------------------------------------------------------------------------------'''
def main ():
    p = Planner()
    p.train()
    rospy.init_node('path_planning')
    p.explore()
    if status.VERBOSE:
        print('Final del rescate. ¡Gracias por usar a ARTU-R!')

if __name__ == '__main__':
    main()