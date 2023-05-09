'''
    Recorta una imagen, quedándose con su parte central y detecta en ella cosas con Yolo
    Jorge F. García-Samartín
    www.gsamartin.es
    21-04-2022
'''
import cv2
import os
import numpy as np
from datetime import datetime
from detect_jorge import (parse_opt, Detection)

def main():
    # Directorio de trabajo
    direccion = os.path.dirname(os.path.abspath(__file__)) + '/'

    # Ventana de imagen con la que nos quedamos
    ventana = 450

    # Leer la imagen
    img = cv2.imread(direccion + 'data/images/bus.jpg')

    # Recortar la imagen
    width = img.shape[1]
    imgCropped = img[:, int(width/2 - ventana/2):int(width/2 + ventana/2)]

    # Guardar la imagen
    fileName = direccion + 'data/images/tests/' + datetime.now().strftime('%Y-%m-%d-%H-%M-%S') + '-bus.jpg'
    cv2.imwrite(fileName, imgCropped)

    # Configuramos las opciones del detector
    opciones = parse_opt()
    opciones.nosave = True
    opciones.save_txt = False
    opciones.save_conf = False
    opciones.source = fileName

    # Detectar en la imagen recortada cosas
    opciones.weights = direccion + 'weights/yolov5s.pt'
    det = Detection(opt=opciones)
    det.detect()

    # SI guardar es muy ineficiente, habría que sacar los rectángulos de la imagen.
    # Habría que meter mano a detect_jorge.py, línea 224
if __name__ == '__main__':
    main()