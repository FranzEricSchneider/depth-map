#!/usr/bin/env python
import sys
import rospy
import cv2
from std_msgs.msg import String
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError

class ImageConverter(object):
  def __init__(self):
    self.image_pub = rospy.Publisher("video_topic", Image, queue_size=10)
    self.bridge = CvBridge()
    self.cap = cv2.VideoCapture(0)

  def read_frame(self):
    return_val, frame = self.cap.read()
    cv2.imshow('frame', frame)
    return frame
    
  def publish_ros_img(self, frame)
    try:
      rosimg = self.bridge.cv2_to_imgmsg(frame, "bgr8")
      self.image_pub.publish(rosimg)
    except CvBridgeError, e:
      print e

def main(args):
  ic = ImageConverter()
  rospy.init_node('video_publisher', anonymous=True)
  rospy.loginfo("Press 'q' to quit WHEN FOCUSED ON CV WINDOW")
  try:
    while(True):
      frame = ic.read_frame()
      ic.publish_ros_img(frame)
      if cv2.waitKey(1) & 0xFF == ord('q'):
        break
  except KeyboardInterrupt:
    print "Shutting down"
  ic.cap.release()
  cv2.destroyAllWindows()


if __name__ == '__main__':
    main(sys.argv)
