#!/usr/bin/env python
import cv2
from glob import glob
from time import sleep

camera = 1
folder = "library_translation_in_L"

cap = cv2.VideoCapture(camera)
test_imgs = []
good = 0
to_capture = 20
sleep(1)
while good <= to_capture:
    ret, frame = cap.read()
    cv2.imshow('frame', frame)
    cv2.waitKey(25)
    fname = '../images/%s/img_%d.jpg' %(folder, good)
    cv2.imwrite(fname, frame)
    print "I just wrote img_%d to %s" %(good, fname)
    sleep(0.25)
    good +=1

cv2.destroyWindow('frame')
print "I just captured %d images!" %to_capture
