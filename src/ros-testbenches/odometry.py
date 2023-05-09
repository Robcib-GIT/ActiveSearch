#!/usr/bin/env python
'''
    Publica topic ficticio de odometría
    Jorge F. García-Samartín
    www.gsamartin.es
    06-05-2022
'''
import rospy
from nav_msgs.msg import Odometry

'''-----------------------------------------------------------------------------------------------
    Publica una odometría ficticia en el topic /odom
-----------------------------------------------------------------------------------------------'''
def odometry():
    pub = rospy.Publisher('/odom', Odometry, queue_size=10)
    rate = rospy.Rate(1)
    while not rospy.is_shutdown():
        print('Nueva odometría recibida')
        odom = Odometry()
        odom.header.stamp = rospy.Time.now()
        odom.header.frame_id = 'odom'
        odom.child_frame_id = 'base_link'
        odom.pose.pose.position.x = -0.05997299994441036
        odom.pose.pose.position.y = 0.001771413299459775
        odom.pose.pose.position.z = 0
        odom.pose.pose.orientation.x = 0
        odom.pose.pose.orientation.y = 0
        odom.pose.pose.orientation.z = -0.014330214206546045
        odom.pose.pose.orientation.w = 1
        odom.twist.twist.linear.x = 0
        odom.twist.twist.linear.y = 0
        odom.twist.twist.linear.z = 0
        odom.twist.twist.angular.x = 0
        odom.twist.twist.angular.y = 0
        odom.twist.twist.angular.z = 0
        pub.publish(odom)
        rate.sleep()

'''-----------------------------------------------------------------------------------------------
    Main
-----------------------------------------------------------------------------------------------'''
def main():
    rospy.init_node('odometry_publisher')
    try:
        odometry()
    except rospy.ROSInterruptException:
        print('Error')
        pass

if __name__ == '__main__':
    main()