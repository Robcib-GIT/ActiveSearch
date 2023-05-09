#!/usr/bin/env python
'''
    Pruebas de movimiento del robot
    Jorge F. García-Samartín
    www.gsamartin.es
    20-05-2022
'''
import os
import rospy
from nav_msgs.msg import Odometry 
from robotControl import Robot
import status

def main():
    # Inicialización del nodo
    rospy.init_node('pruebas_mov', anonymous = True)
    # Inicialización del robot
    robot = Robot(dir = os.path.dirname(os.path.abspath(__file__)) + '/')
    print('a')
    rospy.wait_for_message('/odom', Odometry)
    print('Odometría recibida')
    robot.readPos()
    x = robot.x
    y = robot.y
    w = robot.w
    # Parámetros de la prueba
    distX = 1.1
    distY = 0
    ang = 0.01
    # Prueba de movimiento
    print(x + distX, y + distY, w + ang)
    robot.moveTo(x + distX, y + distY, w + ang)
    print('Hecho1')
    robot.readPos()
    x = robot.x
    y = robot.y
    w = robot.w
    print(x,y,w)
    # Espera
    rospy.sleep(10)
    # Prueba de movimiento
    robot.readPos()
    x = robot.x
    y = robot.y
    w = robot.w
    print(x - distX, y - distY, w - ang)
    robot.moveTo(x - distX, y - distY, w - ang)
    print('Hecho2')
    # Espera
    rospy.sleep(0.5)

if __name__ == '__main__':
    main()