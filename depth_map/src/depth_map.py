#!/usr/bin/env python
import rospy
import cv2
import pickle
import numpy as np
from math import exp
import os

import rospkg
import tf
from sensor_msgs.msg import PointCloud
from std_msgs.msg import Header
from geometry_msgs.msg import Point32

# Code adopted from https://github.com/paulruvolo/comprobo2014/blob/master/exercises/epipolar_geometry/show_depth_auto.py

# TODO: class conversions? 

class DepthMap(object):
    """DepthMap! Not a death map."""
    def __init__(self, cam_path, img1_path, img2_path):
        """
        Creates a depth map object! dundundun.

        Inputs:
        * cam: (string) filepath to the appropriate pickle that contains 
               the calibration matrices
        * img1: (string) filepath for image 1
        * img2: (string) filepath for image 2
        """
        super(DepthMap, self).__init__()

        self.detector = cv2.FeatureDetector_create('SIFT')
        self.extractor = cv2.DescriptorExtractor_create('SIFT')
        self.matcher = cv2.BFMatcher()

        pack = rospkg.RosPack()
        package_path = pack.get_path("depth_map")
        self.img1_path = os.path.join(package_path, "images", img1_path)
        self.img2_path = os.path.join(package_path, "images", img2_path)
        cam_path = os.path.join(package_path, "cameras", cam_path)

        cam = pickle.load( open( cam_path , 'rb'  ) )
        self.D = cam[1]
        self.K = cam[0]
        self.W = np.array([[0.0, -1.0, 0.0],
                           [1.0, 0.0, 0.0],
                           [0.0, 0.0, 1.0]])

        self.corner_threshold = 0.0
        self.ratio_threshold = 1.0
        self.epipolar_threshold = 0.006737946999085467

        self.colors = [(255,0,0),(0,255,0),(0,0,255),(255,255,0),(255,0,255),(0,255,255)]
        self.pt_num = 0
        self.im1_pts = []
        self.im2_pts = []
        # self.show_lines = True # draws lines b/t correspondences
        self.show_lines = False # draws lines b/t correspondences

        self.F = np.zeros([3,4])
        self.E = np.zeros([3,4])

        # the first camera has a camera matrix with no translation or rotation
        self.P = np.array([[1.0, 0.0, 0.0, 0.0],
                          [0.0, 1.0, 0.0, 0.0],
                          [0.0, 0.0, 1.0, 0.0]])

        rospy.init_node('depth_publisher')
        self.tf_broadcaster = tf.TransformBroadcaster()
        self.point_pub = rospy.Publisher('points', PointCloud, queue_size=10)

    def triangulate_points(self, pt_set1, pt_set2, P1):
        my_points = cv2.triangulatePoints(self.P, P1, pt_set1.T, pt_set2.T)
        projected_points_1 = self.P.dot(my_points)
        
        # convert to inhomogeneous coordinates
        for i in range(projected_points_1.shape[1]):
            projected_points_1[0,i] /= projected_points_1[2,i]
            projected_points_1[1,i] /= projected_points_1[2,i]
            projected_points_1[2,i] /= projected_points_1[2,i]

        projected_points_2 = P1.dot(my_points)
        # convert to inhomogeneous coordinates
        for i in range(projected_points_2.shape[1]):
            projected_points_2[0,i] /= projected_points_2[2,i]
            projected_points_2[1,i] /= projected_points_2[2,i]
            projected_points_2[2,i] /= projected_points_2[2,i]

        # convert to inhomogeneous coordinates
        for i in range(projected_points_2.shape[1]):
            my_points[0,i] /= my_points[3,i]
            my_points[1,i] /= my_points[3,i]
            my_points[2,i] /= my_points[3,i]
            my_points[3,i] /= my_points[3,i]

        return my_points.T

    def test_epipolar(self, pt1, pt2):
        pt1_h = np.zeros((3,1))
        pt2_h = np.zeros((3,1))
        pt1_h[0:2,0] = pt1.T
        pt2_h[0:2,0] = pt2.T
        pt1_h[2] = 1.0
        pt2_h[2] = 1.0
        return pt2_h.T.dot(self.E).dot(pt1_h)

    def test_triangulation(self, P, pcloud):
        P4x4 = np.eye(4)
        P4x4[0:3, :] = P
        pcloud_3d = pcloud[:,0:3]
        projected = cv2.perspectiveTransform(np.array([pcloud_3d]),P4x4)
        return np.mean(projected[0,:,2]>0.0)

    def mouse_event(self, event, x, y, flag, dc):
        if event == cv2.EVENT_FLAG_LBUTTON:
            if x < im.shape[1]/2.0:
                l = self.F.dot(np.array([x,y,1.0]))
                m = -l[0]/l[1]
                b = -l[2]/l[1]
                # equation of the line is y = m*x+b
                y_for_x_min = m*0.0+b
                y_for_x_max = m*(im.shape[1]/2.0-1)+b
                # plot the epipolar line
                cv2.line(im,(int(im.shape[1]/2.0),int(y_for_x_min)),(int(im.shape[1]-1.0),int(y_for_x_max)),(255,0,0))

    def compute_depths(self, im1, im2, im1_bw, im2_bw):
        global im

        kp1 = self.detector.detect(im1_bw)
        kp2 = self.detector.detect(im2_bw)

        dc, des1 = self.extractor.compute(im1_bw, kp1)
        dc, des2 = self.extractor.compute(im2_bw, kp2)

        # do matches both ways so we can better screen out spurious matches
        matches = self.matcher.knnMatch(des1, des2, k = 2)
        matches_reversed = self.matcher.knnMatch(des2, des1, k = 2)

        # apply the ratio test in one direction
        good_matches_prelim = []
        for m,n in matches:
            if m.distance < self.ratio_threshold * n.distance and \
               kp1[m.queryIdx].response > self.corner_threshold and \
               kp2[m.trainIdx].response > self.corner_threshold:
                good_matches_prelim.append((m.queryIdx, m.trainIdx))

        # apply the ratio test in the other direction
        good_matches = []
        for m,n in matches_reversed:
            if m.distance < self.ratio_threshold * n.distance and \
               (m.trainIdx, m.queryIdx) in good_matches_prelim:
                good_matches.append((m.trainIdx, m.queryIdx))

        if len(good_matches) == 0:
            print "No good matches, yo!"
            return None

        auto_pts1 = np.zeros((1, len(good_matches), 2))
        auto_pts2 = np.zeros((1, len(good_matches), 2))

        for idx in range(len(good_matches)):
            match = good_matches[idx]
            auto_pts1[0, idx, :] = kp1[match[0]].pt
            auto_pts2[0, idx, :] = kp2[match[1]].pt

        auto_pts1_orig = auto_pts1
        auto_pts2_orig = auto_pts2

        print auto_pts1

        # remove the effect of the intrinsic parameters as well as radial distortion
        auto_pts1 = cv2.undistortPoints(auto_pts1, self.K, self.D)
        auto_pts2 = cv2.undistortPoints(auto_pts2, self.K, self.D)

        print auto_pts1

        correspondences = [[],[]]
        for i in range(auto_pts1_orig.shape[1]):
            correspondences[0].append((auto_pts1_orig[0,i,0],auto_pts1_orig[0,i,1]))
            correspondences[1].append((auto_pts2_orig[0,i,0],auto_pts2_orig[0,i,1]))

        im1_pts = np.zeros((len(correspondences[0]),2))
        im2_pts = np.zeros((len(correspondences[1]),2))

        # usage of global im
        im = np.array(np.hstack((im1, im2)))

        # plot the points
        for i in range(len(im1_pts)):
            im1_pts[i,0] = correspondences[0][i][0]
            im1_pts[i,1] = correspondences[0][i][1]
            im2_pts[i,0] = correspondences[1][i][0]
            im2_pts[i,1] = correspondences[1][i][1]

            cv2.circle(im, (int(im1_pts[i, 0]), int(im1_pts[i, 1])), 2,
                       (255, 0, 0), 2)
            cv2.circle(im, (int(im2_pts[i, 0] + im1.shape[1]),
                       int(im2_pts[i, 1])), 2, (255, 0, 0), 2)

        # the np.array bit makes the points into a 1xn_pointsx2 numpy array since that is what undistortPoints requires
        # TODO; DK : we haven't confirmed the undistorted points... plot them somehow?
        im1_pts_ud = cv2.undistortPoints(np.array([im1_pts]),self.K,self.D)
        im2_pts_ud = cv2.undistortPoints(np.array([im2_pts]),self.K,self.D)

        # since we are using undistorted points we are really computing the essential matrix
        # TODO; DK : check E?
        self.E, mask = cv2.findFundamentalMat(im1_pts_ud, im2_pts_ud,
                                              cv2.FM_RANSAC,
                                              self.epipolar_threshold)

        # correct matches using the optimal triangulation method of Hartley and Zisserman
        # TODO; DK : check corrected pts,,, also plot these somehow? plot im1_pts_ud with lines, and im_pts_fixed with differently colored lines
        im1_pts_ud_fixed, im2_pts_ud_fixed = cv2.correctMatches(self.E,
                                                                im1_pts_ud,
                                                                im2_pts_ud)

        M, mask = cv2.findHomography(auto_pts1_orig, auto_pts2_orig, cv2.RANSAC)
        h, w, c = im1.shape
        pts = np.float32([ [20,20], [20, h-21], [w-21, h-21], [w-21, 20] ]).reshape(-1, 1, 2)
        dst = cv2.perspectiveTransform(pts,M)

        # visualizing some homography things!
        # print pts
        # print dst
        for pt in pts:
            cv2.circle(im, (pt[0,0], pt[0,1]), 2, (0, 0, 255) , 20)
        for pt in dst:
            cv2.circle(im, (int(pt[0,0])+w, int(pt[0,1])), 2, (0, 255, 0) , 20)

        for i in range(4):
            if not i == 3:
                cv2.line(im, (pts[i,0,0],pts[i,0,1]), (pts[i+1,0,0],pts[i+1,0,1]), (0,0,255), 2)
                cv2.line(im, (int(dst[i,0,0])+w,dst[i,0,1]), (int(dst[i+1,0,0])+w,dst[i+1,0,1]), (0,255,0), 2)
            else:
                cv2.line(im, (pts[i,0,0],pts[i,0,1]), (pts[0,0,0],pts[0,0,1]), (0,0,255), 2)
                cv2.line(im, (int(dst[i,0,0])+w,dst[i,0,1]), (int(dst[0,0,0])+w,dst[0,0,1]), (0,255,0), 2)


        epipolar_error = np.zeros((im1_pts_ud_fixed.shape[1],))
        for i in range(im1_pts_ud_fixed.shape[1]):
            epipolar_error[i] = self.test_epipolar(im1_pts_ud_fixed[0, i, :],
                                                   im2_pts_ud_fixed[0, i, :])

        # since we used undistorted points to compute F we really computed E, now we use E to get F
        # F is just for display funsies...
        self.F = np.linalg.inv(self.K.T).dot(self.E).dot(np.linalg.inv(self.K))
        U, Sigma, V = np.linalg.svd(self.E)

        # these are the two possible rotations
        # only need R for the sake of finding P1
        R1 = U.dot(self.W).dot(V)
        R2 = U.dot(self.W.T).dot(V)

        # flip sign of E if necessary
        if np.linalg.det(R1)+1.0 < 10**-8:
            # flip sign of E and recompute everything
            self.E = -self.E
            self.F = np.linalg.inv(self.K.T).dot(self.E).dot(np.linalg.inv(self.K))
            U, Sigma, V = np.linalg.svd(self.E)

            R1 = U.dot(self.W).dot(V)
            R2 = U.dot(self.W.T).dot(V)

        # these are the two possible translations between the two cameras (up to a scale)
        t1 = U[:,2]
        t2 = -U[:,2]

        P1_possibilities = [np.column_stack((R1, t1)),
                            np.column_stack((R1, t2)),
                            np.column_stack((R2, t1)),
                            np.column_stack((R2, t2))]

        pclouds = []
        for P1 in P1_possibilities:
            pclouds.append(self.triangulate_points(im1_pts_ud_fixed,
                                                   im2_pts_ud_fixed, P1))

        # compute the proportion of points in front of the cameras
        infront_of_camera = []
        for i in range(len(P1_possibilities)):
            infront_of_camera.append(self.test_triangulation(self.P, pclouds[i]) + \
                                     self.test_triangulation(P1_possibilities[i], pclouds[i]))

        # the highest proportion of points in front of the cameras is the one we select
        best_pcloud_idx = np.argmax(infront_of_camera)
        # TODO; DK : check P1?? some test calculations (for when we know what translation/rotation we used in real life)

        best_pcloud = pclouds[best_pcloud_idx]

        # scale the depths between 0 and 1 so it is easier to visualize
        depths = best_pcloud[:, 2] - min(best_pcloud[:, 2])
        depths = depths / max(depths)

        return best_pcloud, depths, im1_pts, im2_pts

    def publish_points(self, pcloud, seq):
        pc = PointCloud()
        pc.header = Header(seq, rospy.Time.now(), 'points')
        for i in range(len(pcloud[:, 0])):
            # Publishing as (z, x, y) b/c of the way that Z is forward in
            # triangulation
            pc.points.append(Point32(pcloud[i, 2],
                                     pcloud[i, 0],
                                     pcloud[i, 1]))
        self.point_pub.publish(pc)
        self.tf_broadcaster.sendTransform((0, 0, 0),
                 tf.transformations.quaternion_from_euler(0, 0, 0),
                 rospy.Time.now(),
                 'points',
                 'map')

    def set_corner_threshold(self, thresh):
        """ Sets the threshold to consider an interest point a corner.  The higher the value
            the more the point must look like a corner to be considered """
        self.corner_threshold = thresh/1000.0

    def set_ratio_threshold(self, thresh):
        """ Sets the ratio of the nearest to the second nearest neighbor to consider the match a good one """
        self.ratio_threshold = thresh/100.0

    def set_epipolar_threshold(self, thresh):
        """ Sets the maximum allowable epipolar error to be considered an inlier by RANSAC """
        self.epipolar_threshold = exp(-10 + thresh / 10.0)

    def run(self):
        im1 = cv2.imread(self.img1_path)
        im2 = cv2.imread(self.img2_path)

        if im1 == None or im2 == None:
            print "One of the file paths was empty"
            return

        im1_bw = cv2.cvtColor(im1, cv2.COLOR_BGR2GRAY)
        im2_bw = cv2.cvtColor(im2, cv2.COLOR_BGR2GRAY)

        cv2.namedWindow("MyWindow")

        # compute the 3-d cooredinates and scaled depths for visualization
        best_pcloud, depths, im1_pts, im2_pts = self.compute_depths(im1, im2, im1_bw, im2_bw)

        # the mouse callback will be used for epipolar lines
        cv2.setMouseCallback("MyWindow", self.mouse_event, im)

        # create a simple UI for setting corner and ratio thresholds
        cv2.namedWindow('UI')
        cv2.createTrackbar('Corner Threshold', 'UI', 0, 100,
                           self.set_corner_threshold)
        cv2.createTrackbar('Ratio Threshold', 'UI', 100, 100,
                           self.set_ratio_threshold)
        cv2.createTrackbar('Epipolar Error Threshold', 'UI', 50, 100,
                           self.set_epipolar_threshold)

        print "hit spacebar to recompute depths"
        counter = 0
        while True:
            try:
                for i in range(best_pcloud.shape[0]):
                    cv2.circle(im, (int(im1_pts[i, 0]), int(im1_pts[i, 1])),
                               int(max(1.0, depths[i] * 20.0)), (0, 255, 0), 1)

                if self.show_lines:
                    for i in range(len(im1_pts)):
                        cv2.line(im,(int(im1_pts[i][0]), int(im1_pts[i][1])),(640 + int(im2_pts[i][0]), int(im2_pts[i][1])),(255,0,0))

                cv2.imshow("MyWindow", im)
                self.publish_points(best_pcloud, counter)

                key = cv2.waitKey(50)
                if key & 0xFF == ord('q'):
                    break
                elif key != -1 and key & 0xFF == ord(' '):
                    best_pcloud, depths, im1_pts, im2_pts = self.compute_depths(im1, im2, im1_bw, im2_bw)
                counter += 1
            except Exception, e:
                print e
                print "Something was probably empty"


        cv2.destroyAllWindows()

if __name__ == "__main__":
    cam_path = 'lindsey_cam.p'
    img1_path = 'ac_126_floor_L/img_30.jpg'
    img2_path = 'ac_126_floor_L/img_25.jpg'
    dm = DepthMap(cam_path, img1_path, img2_path)
    dm.run()
