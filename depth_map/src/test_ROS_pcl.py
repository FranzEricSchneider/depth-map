#!/usr/bin/env python
import rospy
from sensor_msgs.msg import PointCloud
from std_msgs.msg import Header
from geometry_msgs.msg import Point32
import tf

br = tf.TransformBroadcaster()

def talker(PC):
    pub = rospy.Publisher('points', PointCloud, queue_size=10)
    r = rospy.Rate(10)
    counter = 0
    while not rospy.is_shutdown():
        PC.header.seq = counter
        PC.header.stamp = rospy.Time.now()
        counter += 1

        pub.publish(PC)
        br.sendTransform((0, 0, 0),
                 tf.transformations.quaternion_from_euler(0, 0, 0),
                 rospy.Time.now(),
                 'points',
                 'map')
        r.sleep()

if __name__ == '__main__':
    rospy.init_node('talker', anonymous=True)

    P = PointCloud()
    P.header = Header(0, 0, 'points')
    P.points.append(Point32(2, 0, 0)); P.points.append(Point32(3, 0, 0))
    P.points.append(Point32(4, 0, 0)); P.points.append(Point32(5, 0, 0))

    P.points.append(Point32(0, 3, 0)); P.points.append(Point32(0, 6, 0))
    P.points.append(Point32(0, 6.5, 0)); P.points.append(Point32(0, 7, 0))

    P.points.append(Point32(0, 0, 2)); P.points.append(Point32(0, 0, -2))
    P.points.append(Point32(0, 0, 1)); P.points.append(Point32(0, 0, -1))
    # for i in range(2):
    #     P.points.append(Point32(i, i, i))

    talker(P)
