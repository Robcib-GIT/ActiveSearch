#!/usr/bin/env python
'''
    Pruebas de movimiento del robot con el topic move_base/goal
    Jorge F. García-Samartín
    www.gsamartin.es
    23-05-2022
'''
import time
import rospy
from nav_msgs.msg import Odometry 
from geometry_msgs.msg import Twist
from move_base_msgs.msg import MoveBaseActionGoal
from robotControl import Robot
import status

vel_lin_A1 = 0
vel_ang_A1 = 0 

def callback_vel_to_A1(msg):
    global vel_lin_A1, vel_ang_A1 
    print('callback')

    vel_lin_A1 = msg.linear.x
    vel_ang_A1 = msg.angular.z

def main():
    global vel_lin_A1, vel_ang_A1 

    # Inicialización del nodo
    rospy.init_node('pruebas_goal', anonymous = True)
    print('Hola')
    rospy.Subscriber("/cmd_vel_A1", Twist, callback_vel_to_A1)  # para el A1
    pub_vel = rospy.Publisher('/cmd_vel', Twist, queue_size = 10)

    # Inicialización del robot
    robot = Robot()
    print('a')
    rospy.wait_for_message('/odom', Odometry)
    print('Odometría recibida')
    robot.readPos()
    x = robot.x
    y = robot.y
    w = robot.w

    # Parámetros de la prueba
    distX = -0.1
    distY = 0

    goal = MoveBaseActionGoal()
    goal.goal.target_pose.pose.position.x = x + distX
    goal.goal.target_pose.pose.position.y = y + distY
    goal.goal.target_pose.pose.orientation.w = w

    pub_goal = rospy.Publisher('move_base/goal', MoveBaseActionGoal, queue_size = 10)
    pub_goal.publish(goal)


    twist = Twist()                         
    twist.linear.x = 0
    twist.angular.z = 0 
    pub_vel.publish(twist)
    time.sleep(1)

    # PUBLICAMOS LAS VELOCIDADES
    rospy.wait_for_message('/cmd_vel_A1', Twist)
    vel_lin = vel_lin_A1
    vel_ang = vel_ang_A1
    print(vel_ang, vel_lin)

    twist = Twist()

    twist.linear.x = vel_lin          
    twist.linear.y = 0 
    twist.linear.z = 0         

    twist.angular.x = 0 
    twist.angular.y = 0 
    twist.angular.z = vel_ang
    pub_vel.publish(twist)


    print('Final')

if __name__ == '__main__':
    main()