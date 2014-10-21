#!/usr/bin/env python
import cv2
from glob import glob
import numpy as np
import pickle


X_PTS = 6
Y_PTS = 8


def grab_calib_pics(camera = 1):
    """Walks through getting callibration matrix for a camera"""

    cap = cv2.VideoCapture(camera)
    test_imgs = []
    good = 0
    while good <=15:
        while(True):
            ret, frame = cap.read()
            cv2.imshow('test', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
        ret, corners = cv2.findChessboardCorners(gray, (Y_PTS, X_PTS), None)
        if corners:
            for corner in corners[:, 0, :]:
                cv2.circle(gray, (int(corner[0]), int(corner[1])),
                          10, [0, 0, 255], 2)
            cv2.imshow('gray', gray)
            cv2.waitKey(0)
            cv2.destroyWindow('gray')
            if ret:
                cv2.imwrite('../images/img_%d.jpg' %good, frame)
                print "I just wrote img_%d" %good
                good +=1
        else:
            print "Image was BAD, corners weren't detected"

def calibrate_from_chessboard():
    """
    Assumes that there are at least ten jpg images of the chessboard callibration rig 
    from different views and that these are the only jpg images in this folder

    returns:
    Camera matrix (intrinsic camera parameters)
    Distortion matrix (of camera)
    """
    # prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
    objp = np.zeros((Y_PTS * X_PTS, 3), np.float32)
    objp[:, :2] = np.mgrid[0:Y_PTS, 0:X_PTS].T.reshape(-1, 2)
    
    # Arrays to store object points and image points from all the images.
    objpoints = [] # 3d point in real world space
    imgpoints = [] # 2d points in image plane.
    
    images = glob('../images/img_*.jpg')
    
    if images:
        for fname in images:    
            img = cv2.imread(fname)
            gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
            ret, corners = cv2.findChessboardCorners(gray, (Y_PTS, X_PTS), None)
            
            # If found, add object points, image points (after refining them)
            if ret == True:
                objpoints.append(objp)
                imgpoints.append(corners)

        ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints,
                                                           gray.shape[::-1], 
                                                           None, None)
        return mtx, dist, rvecs, tvecs
    else:
        return None

def check_calibration(mtx, dist):
    alpha = 0
    images = glob('../images/img_*.jpg')
    for fname in images:
        img = cv2.imread(fname)
        h,  w = img.shape[:2]
        newcameramtx, roi = cv2.getOptimalNewCameraMatrix(mtx, dist, (w, h), alpha,
                                                          (w, h))

        # undistort
        dst = cv2.undistort(img, mtx, dist, None, newcameramtx)

        # crop the image
        x, y, w, h = roi
        dst = dst[y:y + h, x:x + w]
        cv2.imshow('undistorted', dst)
        cv2.waitKey(0)
        cv2.destroyWindow('undistorted')


if __name__ == '__main__':
    c = raw_input('Do you want to take more pictures? (y/n) ')
    if c[0] == 'y':
        grab_calib_pics()
    c = raw_input('Do you want to do calibration from taken pics? (y/n) ')
    if c[0] == 'y':
        mtx, dist, rvecs, tvecs = calibrate_from_chessboard()
        c = raw_input('Do you want to save the camera matrix? (y/n) ')
        if c[0] == 'y':
            fname = raw_input('What camera are you saving? (e.g. lindsey_cam)')
            pickle.dump([mtx, dist], open('../cameras/%s.p'%fname, 'wb'))
        c = raw_input('Do you want to check the pics undistorted? (y/n) ')
        if c[0] == 'y':
            check_calibration(mtx, dist)