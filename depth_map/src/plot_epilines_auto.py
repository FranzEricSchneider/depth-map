import cv2
import pickle
import numpy as np

ratio_threshold = 0.7
corner_threshold = 0.02

# D = np.array( [0.08683, -0.28966000000000003, -0.00045000000000000004, -0.00015000000000000001, 0.0])
# K = np.array( [[651.38582, 0.0, 327.26766], [0.0, 650.2441, 242.38098],[ 0.0, 0.0, 1.0]])

# From our camera_calibration pickles
lindsey_cam = pickle.load( open( '../cameras/lindsey_cam.p' , 'rb'  ) )
graham_cam = pickle.load( open( '../cameras/graham_cam.p' , 'rb'  ) )
D = lindsey_cam[1]
print 'D: \n', D
K = lindsey_cam[0]
print 'K: \n', K

W = np.array([[0.0, -1.0, 0.0],
              [1.0, 0.0, 0.0],
              [0.0, 0.0, 1.0]])

colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0), (255, 0, 255),
          (0, 255, 255)]
pt_num = 0
im1_pts = []
im2_pts = []
im1name = 'img_38.jpg'
im2name = 'img_39.jpg'


# Q: What is P and P1?
# Q: What is all of this inhomogenous stuff?
def triangulate_points(pt_set1, pt_set2, P, P1):
    my_points = cv2.triangulatePoints(P, P1, pt_set1.T, pt_set2.T)
    projected_points_1 = P.dot(my_points)
    
    # convert to inhomogeneous coordinates
    for i in range(projected_points_1.shape[1]):
        projected_points_1[0, i] /= projected_points_1[2, i]
        projected_points_1[1, i] /= projected_points_1[2, i]
        projected_points_1[2, i] /= projected_points_1[2, i]

    projected_points_2 = P1.dot(my_points)
    # convert to inhomogeneous coordinates
    for i in range(projected_points_2.shape[1]):
        projected_points_2[0, i] /= projected_points_2[2, i]
        projected_points_2[1, i] /= projected_points_2[2, i]
        projected_points_2[2, i] /= projected_points_2[2, i]

    # convert to inhomogeneous coordinates
    for i in range(projected_points_2.shape[1]):
        my_points[0, i] /= my_points[3, i]
        my_points[1, i] /= my_points[3, i]
        my_points[2, i] /= my_points[3, i]
        my_points[3, i] /= my_points[3, i]

    return my_points.T

def test_epipolar(E,pt1,pt2):
    pt1_h = np.zeros((3, 1))
    pt2_h = np.zeros((3, 1))
    pt1_h[0:2, 0] = pt1.T
    pt2_h[0:2, 0] = pt2.T
    pt1_h[2] = 1.0
    pt2_h[2] = 1.0
    return pt2_h.T.dot(E).dot(pt1_h)

def test_triangulation(P, pcloud):
    P4x4 = np.eye(4)
    P4x4[0:3, :] = P
    pcloud_3d = pcloud[:, 0:3]
    projected = cv2.perspectiveTransform(np.array([pcloud_3d]), P4x4)
    return np.mean(projected[0, :, 2] > 0.0)

def mouse_event(event,x,y,flag,im):
    if event == cv2.EVENT_FLAG_LBUTTON:
        if x < im.shape[1] / 2.0:
            l = F.dot(np.array([x, y, 1.0]))
            m = -l[0] / l[1]
            b = -l[2] / l[1]
            # equation of the line is y = m*x+b
            y_for_x_min = m * 0.0 + b
            y_for_x_max = m * (im.shape[1] / 2.0 - 1) + b
            # plot the epipolar line
            cv2.line(im, (int(im.shape[1] / 2.0), int(y_for_x_min)), 
                     (int(im.shape[1] - 1.0), int(y_for_x_max)), (255, 0, 0))

if __name__ == '__main__':
    im1 = cv2.imread(im1name)
    im2 = cv2.imread(im2name)

    extract_automatic_matches = True
    add_in_auto_to_compute_E = True
    only_use_automatic_matches = True

    if extract_automatic_matches:
        detector = cv2.FeatureDetector_create('SIFT')
        extractor = cv2.DescriptorExtractor_create('SIFT')
        matcher = cv2.BFMatcher()

        im1_bw = cv2.cvtColor(im1, cv2.COLOR_BGR2GRAY)
        im2_bw = cv2.cvtColor(im2, cv2.COLOR_BGR2GRAY)

        # Finds keypoints
        kp1 = detector.detect(im1_bw)
        kp2 = detector.detect(im2_bw)
        # print ' len(kp1)' 
        # print len(kp1)

        # Finds descriptors
        dc, des1 = extractor.compute(im1_bw, kp1)
        dc, des2 = extractor.compute(im2_bw, kp2)
        # print ' des1.shape' 
        # print des1.shape

        # Flann matcher?
        matches = matcher.knnMatch(des1, des2, k=2)
        matches_reversed = matcher.knnMatch(des2, des1, k=2)

        good_matches_prelim = []
        for m, n in matches:
            if m.distance < ratio_threshold * n.distance and \
               kp1[m.queryIdx].response > corner_threshold and \
               kp2[m.trainIdx].response > corner_threshold:
                good_matches_prelim.append((m.queryIdx, m.trainIdx))

        good_matches = []
        for m, n in matches_reversed:
            if m.distance < ratio_threshold * n.distance and \
               (m.trainIdx,m.queryIdx) in good_matches_prelim:
                good_matches.append((m.trainIdx, m.queryIdx))
        print ' len(good_matches)' 
        print len(good_matches)

        auto_pts1 = np.zeros((1, len(good_matches), 2))
        auto_pts2 = np.zeros((1, len(good_matches), 2))

        for idx in range(len(good_matches)):
            match = good_matches[idx]
            auto_pts1[0, idx, :] = kp1[match[0]].pt
            auto_pts2[0, idx, :] = kp2[match[1]].pt

        auto_pts1_orig = auto_pts1
        auto_pts2_orig = auto_pts2

        auto_pts1 = cv2.undistortPoints(auto_pts1, K, D)
        auto_pts2 = cv2.undistortPoints(auto_pts2, K, D)

    f = open('correspondences.pickle','r')
    correspondences = pickle.load(f)
    f.close()

    if only_use_automatic_matches:
        correspondences = [[], []]

    # Automatically compute correspondences
    if add_in_auto_to_compute_E:
        for i in range(auto_pts1_orig.shape[1]):
            correspondences[0].append((auto_pts1_orig[0, i, 0],
                                       auto_pts1_orig[0, i, 1]))
            correspondences[1].append((auto_pts2_orig[0, i, 0],
                                       auto_pts2_orig[0, i, 1]))
    # print ' len(correspondences[0])' 
    # print len(correspondences[0])
    # print ' correspondences[0][0]' 
    # print correspondences[0][0]

    im1_pts = np.zeros((len(correspondences[0]), 2))
    im2_pts = np.zeros((len(correspondences[1]), 2))

    im = np.array(np.hstack((im1, im2)))

    # Plot the points
    for i in range(len(im1_pts)):
        im1_pts[i, 0] = correspondences[0][i][0]
        im1_pts[i, 1] = correspondences[0][i][1]
        im2_pts[i, 0] = correspondences[1][i][0]
        im2_pts[i, 1] = correspondences[1][i][1]

        cv2.circle(im, (int(im1_pts[i, 0]), int(im1_pts[i, 1])), 2,
                   (255, 0, 0), 2)
        cv2.circle(im, (int(im2_pts[i, 0] + im1.shape[1]), int(im2_pts[i, 1])),
                   2, (255, 0, 0), 2)
        # EXAMINE

    # Q: What is im1_pts_augmented?
    im1_pts_augmented = np.zeros((1, im1_pts.shape[0], im1_pts.shape[1]))
    im1_pts_augmented[0, :, :] = im1_pts
    im2_pts_augmented = np.zeros((1, im2_pts.shape[0], im2_pts.shape[1]))
    im2_pts_augmented[0, :, :] = im2_pts

    im1_pts_ud = cv2.undistortPoints(im1_pts_augmented, K, D)
    im2_pts_ud = cv2.undistortPoints(im2_pts_augmented, K, D)

    # Q: Should this be F?
    E, mask = cv2.findFundamentalMat(im1_pts_ud, im2_pts_ud, cv2.FM_RANSAC,
                                     0.5 * 10 ** -3)

    # Q: Why do we correct this?
    im1_pts_ud_fixed, im2_pts_ud_fixed = cv2.correctMatches(E, im1_pts_ud,
                                                            im2_pts_ud)
    # EXAMINE
    use_corrected_matches = True
    if not(use_corrected_matches):
        im1_pts_ud_fixed = im1_pts_ud
        im2_pts_ud_fixed = im2_pts_ud

    epipolar_error = np.zeros((im1_pts_ud_fixed.shape[1],))
    for i in range(im1_pts_ud_fixed.shape[1]):
        epipolar_error[i] = test_epipolar(E, im1_pts_ud_fixed[0, i, :],
                                          im2_pts_ud_fixed[0, i, :])

    # Q: Did Paul switch up E and F?
    F = np.linalg.inv(K.T).dot(E).dot(np.linalg.inv(K))
    
    # Q: Paper: What are all of these guys for?
    U, Sigma, V = np.linalg.svd(E)
    R1 = U.dot(W).dot(V)
    R2 = U.dot(W.T).dot(V)

    if np.linalg.det(R1) + 1.0 < 10**-8:
        # flip sign of E and recompute everything
        E = -E
        F = np.linalg.inv(K.T).dot(E).dot(np.linalg.inv(K))
        U, Sigma, V = np.linalg.svd(E)

        R1 = U.dot(W).dot(V)
        R2 = U.dot(W.T).dot(V)

    # What is t1?
    t1 = U[:,2]
    t2 = -U[:,2]

    # Q: Paper: What is P for?
    P = np.array([[1.0, 0.0, 0.0, 0.0],
                  [0.0, 1.0, 0.0, 0.0],
                  [0.0, 0.0, 1.0, 0.0]]);
    P1_possibilities = []
    P1_possibilities.append(np.column_stack((R1, t1)))
    P1_possibilities.append(np.column_stack((R1, t2)))
    P1_possibilities.append(np.column_stack((R2, t1)))
    P1_possibilities.append(np.column_stack((R2, t2)))

    # P1_possibilities is a list of of four numpy arrays, each with 3x4 floats
    # print ' P1_possibilities' 
    # print P1_possibilities

    pclouds = []
    for P1 in P1_possibilities:
        pclouds.append(triangulate_points(im1_pts_ud_fixed, im2_pts_ud_fixed,
                                          P, P1))
    # What specifically is in pclouds[0]? Are they 3D? 2D? Which camera are
    # they from?

    infront_of_camera = []
    for i in range(len(P1_possibilities)):
        infront_of_camera.append(test_triangulation(P, pclouds[i]) + 
                                 test_triangulation(P1_possibilities[i],
                                 pclouds[i]))
    best_pcloud_idx = np.argmax(infront_of_camera)
    
    print 'P1_possibilities[best_pcloud_idx]'
    print P1_possibilities[best_pcloud_idx]

    if not(extract_automatic_matches):
        best_pcloud = pclouds[best_pcloud_idx]
    else:
        auto_pts1_fixed, auto_pts2_fixed = cv2.correctMatches(E, auto_pts1,
                                                              auto_pts2)
        pcloud_auto = triangulate_points(auto_pts1_fixed, auto_pts2_fixed, P,
                                         P1_possibilities[best_pcloud_idx])
        best_pcloud = np.vstack((pclouds[best_pcloud_idx], pcloud_auto))
        print ' auto_pts1_orig[0, :, :].shape' 
        print auto_pts1_orig[0, :, :].shape
        
        # Q: Why is im1_pts a vstack of the corrected and uncorrected pts?
        im1_pts = np.vstack((im1_pts, auto_pts1_orig[0,:,:]))
        im2_pts = np.vstack((im2_pts, auto_pts2_orig[0,:,:]))

    depths = best_pcloud[:, 2] - min(best_pcloud[:, 2])
    depths = depths / max(depths)

    for i in range(best_pcloud.shape[0]):
        cv2.circle(im, (int(im1_pts[i, 0]), int(im1_pts[i, 1])),
                   int(max(1.0, depths[i] * 20.0)), (0, depths[i] * 255, 0), 1)

    cv2.imshow(' MYWIN' ,im)
    cv2.setMouseCallback(' MYWIN' ,mouse_event,im)
    while True:
        cv2.imshow(' MYWIN' ,im)
        k = cv2.waitKey(50)
        if k & 0xFF == ord('q'):
            break
    cv2.destroyAllWindows()