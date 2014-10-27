#!/usr/bin/env python
import depth_map
from random import random

if __name__== '__main__':
	camera = 'lindsey_cam.p'
	folder = 'ac_126_box_L'
	for i in range(10):
		num1 = int(random()*50)
		num2 = int(random()*50)
		f1 = '%s/img_%d.jpg' %(folder, num1)
		f2 = '%s/img_%d.jpg' %(folder, num2)
		print f1
		print f2
		dm = depth_map.DepthMap(camera, f1, f2)
		dm.run()
