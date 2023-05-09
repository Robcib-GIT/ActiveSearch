#!/usr/bin/env python
'''
    Funciones para el manejo del movimiento del robot
    Jorge F. García-Samartín
    www.gsamartin.es
    12-05-2022
'''
import rospy
import math
import numpy as np
import time

from datetime import datetime
from geometry_msgs.msg import Twist, PoseStamped
from nav_msgs.msg import Odometry, Path
from move_base_msgs.msg import MoveBaseActionGoal
from scipy import interpolate

import status

# Clase con los parámetros del PID
class PID:

    def __init__(self, kp = 0, ki = 0, kd = 0, tau = 0):
        self.kp = kp
        self.ki = ki   
        self.kd = kd
        self.tau = tau

# Clase que gestiona el movimiento del robot
class Robot:

    def __init__(self, dir=''):
        self.x = 0
        self.y = 0
        self.w = 0
        self.velV = 0
        self.velW = 0
        self.maxDistLin = 0.7
        self.minDistLin = 0.1
        self.tolAng = 0.3
        self.dir = dir

        # Parámetros del PID (Kp, Ki, Kd, Tau)
        self.pid = PID(2, 0.2, 0, 0.1)

        # Para guardar un histórico de sus posiciones
        self.histPos = []   # Posiciones del robot
        self.theorPos = []  # Posiciones teóricas
        self.lastDate = datetime.now().strftime('%Y-%m-%d-%H-%M-%S') 

        # Suscripción a la odometría
        self.subOdom = rospy.Subscriber('/odom', Odometry, self.callbackOdom)
        self.subVel = rospy.Subscriber('/cmd_vel_A1', Twist, self.callbackVel)

        # Publicadores de objetivo y velocidad
        self.pub_traj = rospy.Publisher('/move_base/goal', MoveBaseActionGoal, queue_size=1, latch=True)
        self.pub_vel = rospy.Publisher('/cmd_vel', Twist, queue_size=10)

        # Publciadores de trayectoria y trayectoria discretizada
        self.path_pub = rospy.Publisher('/path', Path, queue_size=10)
        self.path_pub_split = rospy.Publisher('/path_split', Path, queue_size=10)
    
    '''-----------------------------------------------------------------------------------------------
        Callback del topic /odom del robot
    -----------------------------------------------------------------------------------------------'''
    def callbackOdom(self, odom):
        self.odom = odom
        self.readPos()
    
    '''-----------------------------------------------------------------------------------------------
        Callback del topic /cmd_vel_A1 del robot
    -----------------------------------------------------------------------------------------------'''
    def callbackVel(self, vel):
        self.velV = vel.linear.x
        self.velW = vel.angular.z
    
    '''-----------------------------------------------------------------------------------------------
       Convierte un ángulo al intervalo [-pi, pi]
    -----------------------------------------------------------------------------------------------'''
    def wrapToPi(self, alpha):
        return math.remainder(alpha, 2 * math.pi)
    
    '''-----------------------------------------------------------------------------------------------
       Pasa de cuaternios a ángulos de Euler. Devuelve giro en z
    -----------------------------------------------------------------------------------------------'''
    def quat2euler(self, q):
        t3 = +2.0 * (q.w * q.z + q.x * q.y)
        t4 = +1.0 - 2.0 * (q.y * q.y + q.z * q.z)
        return math.atan2(t3, t4)
    
    '''-----------------------------------------------------------------------------------------------
       Pasa de ángulos de Euler a cuaternios
    -----------------------------------------------------------------------------------------------'''
    def euler2quat(self, a):
        w = math.cos(a[0])*math.cos(a[1])*math.cos(a[2]) - math.sin(a[0])*math.sin(a[1])*math.sin(a[2])
        x = math.sin(a[0])*math.sin(a[1])*math.cos(a[2]) + math.cos(a[0])*math.cos(a[1])*math.sin(a[2])
        y = math.sin(a[0])*math.cos(a[1])*math.cos(a[2]) + math.cos(a[0])*math.sin(a[1])*math.sin(a[2])
        z = math.cos(a[0])*math.sin(a[1])*math.cos(a[2]) - math.sin(a[0])*math.cos(a[1])*math.sin(a[2])

        return w, x, y, z
    
    '''-----------------------------------------------------------------------------------------------
        Lee la odometría del robot
    -----------------------------------------------------------------------------------------------'''
    def readPos(self):
        self.pos = self.odom.pose.pose.position
        self.x = self.pos.x
        self.y = self.pos.y
        self.w = self.quat2euler(self.odom.pose.pose.orientation)
    
    '''-----------------------------------------------------------------------------------------------
        Guarda el histórico de posiciones del robot
    -----------------------------------------------------------------------------------------------'''
    def savePos(self):
        self.lastDate = datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
        self.theorPos.extend([[0,0,0]])
        self.theorPos.extend(self.histPos)
        np.savetxt(self.dir + 'results/low/' + self.lastDate + '-histPos.csv', self.theorPos, delimiter=',', fmt='%1.5f')
        self.histPos = []
        self.theorPos = []
    
    '''-----------------------------------------------------------------------------------------------
        Divide una trayectoria en puntos más cercanos
    -----------------------------------------------------------------------------------------------'''
    def spliteTraj (self, points, tangents, resolution):
        resolution = float(resolution)
        points = np.asarray(points)
        nPoints, dim = points.shape
        dp = np.diff(points, axis=0)                 # difference between points
        dp = np.linalg.norm(dp, axis=1)              # distance between points
        d = np.cumsum(dp)                            # cumsum along the segments
        d = np.hstack([[0],d])                       # add distance from first point
        l = d[-1]                                    # length of point sequence
        nSamples = int(l/resolution)                 # number of samples
        s,r = np.linspace(0,l,nSamples,retstep=True) # sample parameter and step
        assert(len(points) == len(tangents))
        data = np.empty([nPoints, dim], dtype=object)
        for i,p in enumerate(points):
            t = tangents[i]
            assert(t is None or len(t)==dim)
            fuse = list(zip(p,t) if t is not None else zip(p,))
            data[i,:] = fuse
        samples = np.zeros([nSamples, dim])
        for i in range(dim):
            poly = interpolate.BPoly.from_derivatives(d, data[:,i])
            samples[:,i] = poly(s)
        return samples

    '''-----------------------------------------------------------------------------------------------
        Movimiento a un punto haciendo path planning
    -----------------------------------------------------------------------------------------------'''
    def moveTo(self, x, y, w_ref = 0, onlyOr = False):

        # Si solo queremos orientarnos, no hay path planning
        if onlyOr == True:
            self.moveToPoint(x, y, w_ref, True)
            return

        # Gestión del path gráfico
        resolution = 0.1
        points_spl = [[self.x, self.y], [x, y]]
        tangents_spl = [[0,1], [0,1]]
        points_spl = np.asarray(points_spl)
        tangents_spl = np.asarray(tangents_spl) 
        scale = 0.5
        tangents1 = np.dot(tangents_spl, scale*np.eye(2))
        trayectoria_discretizada = self.spliteTraj(points_spl, tangents1, resolution)

        path = Path()
        path_split = Path()

        path.header.frame_id = 'map'
        path_split.header.frame_id = 'map'

        for k in range(len(trayectoria_discretizada[:,1])):		
            pose = PoseStamped()
            pose.header.stamp = rospy.Time.now()
            pose.header.frame_id = 'map'
            pose.header.seq = k	
            pose.pose.position.x = trayectoria_discretizada[k,0]
            pose.pose.position.y = trayectoria_discretizada[k,1]
            path_split.poses.append(pose)       

        self.path_pub_split.publish(path_split)

        # Gestión de la trayectoria real
        goal_A1 = MoveBaseActionGoal()
        goal_A1.goal.target_pose.header.frame_id = 'map'	
        goal_A1.goal.target_pose.pose.position.x = x
        goal_A1.goal.target_pose.pose.position.y = y
        goal_A1.goal.target_pose.pose.orientation.w = 1.0

        self.pub_traj.publish(goal_A1)

        time.sleep(1)

        # Publicamos las velocidades
        error_distancia = math.sqrt((self.x - x)**2 + (self.y - y)**2)
        
        while error_distancia > self.maxDistLin:
            
            error_distancia = math.sqrt((self.x - x)**2 + (self.y - y)**2)

            vel_lin = self.velV
            vel_ang = self.velW

            twist = Twist()

            twist.linear.x = vel_lin
            twist.angular.z = vel_ang
            self.pub_vel.publish(twist)
    
    '''-----------------------------------------------------------------------------------------------
        Movimiento en línea recta a un punto (x,y) con una orientación determinada
    -----------------------------------------------------------------------------------------------'''
    def moveToPoint(self, x, y, w_ref = 0, onlyOr = False):
        # Control de salida del bucle
        returnV = False
        returnW = False

        # Errores
        w_ref = self.wrapToPi(w_ref)
        error_or = self.wrapToPi(self.w - w_ref)
        showError = 0

        # Guardamos la posición teórica inicial y la final
        self.theorPos.append([self.x, self.y, self.w])
        self.theorPos.append([x, y, w_ref])

        # 	CORREGIMOS EN ORIENTACION PRIMERO PARA EL TRACKING
        while error_or != 0:

            # Guardamos la posición actual
            self.histPos.append([self.x, self.y, self.w])

		    # AJUSTE Y LIMITACIONS DE VELOCIDADES
            error_orientacion = -1 * error_or
            error_distancia = math.sqrt((self.x - x)**2 + (self.y - y)**2)

		    # CALCULAMOS LAS RESPECTIVAS VELOCIDADES EN FUNCION DE LOS ERRORES
            vel_ang = error_orientacion * 0.1 * self.pid.kp + self.pid.tau*self.pid.ki*error_orientacion
            vel_lin = error_distancia * self.pid.kp + self.pid.tau*self.pid.ki*error_orientacion

            if status.VERBOSE and status.TEST and showError == 1000:
                print(error_distancia, error_orientacion, w_ref, self.w)
                showError = 0
            else:
                showError += 1

            # MAXIMOS Y MINIMOS DE VELOCIDAD
            if vel_ang > 0.6:
                vel_ang = 0.1
                returnW = False
            if vel_ang < -0.6:
                vel_ang = -0.1
                returnW = False
            if vel_ang < 0.07 and vel_ang > -0.07:
                vel_ang = 0
                returnW = True

            if vel_lin > 0.3:
                vel_lin = 0.1
                returnV = False
            if vel_lin < -0.3:
                vel_lin = -0.1
                returnV = False
            if vel_lin < 0.03 and vel_lin > -0.03:
                vel_lin = 0
                returnV = True

            if abs(error_distancia) < self.maxDistLin and abs(error_distancia) > self.minDistLin: 
                vel_lin = 0
                returnV = True
                if status.TEST and status.VERBOSE and not(onlyOr):
                    print('Ha llegado en distancia')

            if abs(error_orientacion) < self.tolAng: # SI LA PERSONA YA ESTA DENTRE DE ESE RANGO DETIENE EL GIRO 
                vel_ang = 0
                returnW = True
                if status.TEST and status.VERBOSE:
                    print('Ha llegado en ángulo')
                    
            if abs(error_orientacion) > self.tolAng:
                vel_lin = 0
                returnV = False

            if abs(error_distancia) < self.minDistLin and abs(error_distancia) != 0:
                vel_lin = -0.2
                returnV = False

            # Si solo queremos orientarnos, no nos movemos
            if onlyOr:
                vel_lin = 0
                returnV = True

		    # PUBLICAMOS LAS VELOCIDADES
            twist = Twist()
            twist.linear.x = vel_lin              
            twist.linear.y = 0 
            twist.linear.z = 0         
            twist.angular.x = 0 
            twist.angular.y = 0 
            twist.angular.z = vel_ang
            if status.TEST_LOW:
                print('Errores lineal y angular: ' + str(error_distancia) + ',' + str(error_orientacion))
                print('Velocidades lineal y angular: ' + str(twist.linear.x) + ',' + str(twist.angular.z))
            pub_vel = rospy.Publisher('cmd_vel', Twist, queue_size = 10)
            pub_vel.publish(twist)

            # Para salirse del bucle
            if returnV and returnW:
                if status.TEST and status.VERBOSE:
                    print('Ha llegado en distancia y ángulo')
                if status.SAVE_POS:
                    self.savePos()
                return

            # Leemos la posición y actualizamos el error
            self.readPos()
            if not onlyOr:
                w_ref = math.atan2(y - self.y, x - self.x)

            error_or = self.wrapToPi(self.w - w_ref)
            if status.TEST_LOW:
                print('Ángulos: ' + str(self.w) + ', ' + str(w_ref))
                print('----------------------------------------------------')

        else:
            if status.TEST and status.VERBOSE:
                print('Error de orientación: ' + str(error_or))
                print('Ángulos: ' + str(self.w) + ', ' + str(w_ref))
                print('Fin del bucle')

                if status.SAVE_POS:
                    self.savePos()