import os
import random
import cv2
import numpy as np
# from scipy.io import loadmat
import matplotlib.pyplot as plt
from matplotlib import cm
from tqdm import tqdm

def ransacF(x1, y1, x2, y2, thresh = 0.001):
  # Find normalization matrix
  # Transform point set 1 and 2
  # RANSAC based 8-point algorithm
  # YOUR CODE HERE: 

  pt1 = normalize(x1, y1)
  pt2 = normalize(x2, y2)

  inliers = []
  F_bank = []
  indices = []

  for i in range(10000):
    rand = np.random.choice(x1.shape[0], 8, replace=False)
    x1R = pt1[:,0][rand]
    y1R = pt1[:,1][rand]
    x2R = pt2[:,0][rand]
    y2R = pt2[:,1][rand]

    F = computeF(x1R, y1R, x2R, y2R)

    inlier, index = getInliers(pt1, pt2, F, thresh)

    inliers.append(inlier)
    indices.append(index)
    F_bank.append(F)

  best = np.argmax(np.array(inliers))
  F_best = F_bank[best]
  indices_best = np.array(indices[best])

  F_best = denormalize(x1, y1, x2, y2, F_best)

  return F_best, indices_best

def getInliers(pt1, pt2, F, thresh):
  # Function: implement the criteria checking inliers. 

  inliers = 0
  indices = []

  for i in range(pt1.shape[0]):
    p1 = pt1[i,:].reshape((1,3))
    p2 = pt2[i,:].reshape((1,3))

    n1 = F @ p2.T
    d1 = np.abs(p1 @ n1)/np.sqrt(n1[0]**2 + n1[1]**2)

    n2 = F.T @ p1.T
    d2 = np.abs(p2 @ n2)/np.sqrt(n2[0]**2 + n2[1]**2)

    score = d1 + d2
    # print(score)

    if score[0,0] < thresh:
      inliers = inliers + 1
      indices.append(i)
    
  # print(inliers)

  return inliers, indices


def normalize(x, y, matrix = False):
  # Function: find the transformation to make it zero mean and the variance as sqrt(2)
  # YOUR CODE HERE:
  og_coord = np.ones((x.shape[0],3))
  og_coord[:,0] = x[:,0]
  og_coord[:,1] = y[:,0]

  mean = (np.mean(x),np.mean(y))
  translate = np.array([[1,0,-mean[0]],
                        [0,1,-mean[1]],
                        [0,0,1]])
  
  coord = np.transpose(np.dot(translate, np.transpose(og_coord)))

  k_x = np.sqrt((x.shape[0])/(np.sum(coord[:,0]**2)))
  k_y = np.sqrt((y.shape[0])/(np.sum(coord[:,1]**2)))

  scaling = np.array([[k_x,0,0],
                      [0,k_y,0],
                      [0,0,1]])
  
  coord = np.transpose(np.dot(scaling, np.transpose(coord)))

  # x_new = coord[:,0]
  # y_new = coord[:,1]

  if matrix:
    T = np.dot(scaling, translate)
    return T
  
  else:
    return coord

  
def computeF(x1, y1, x2, y2):
  #  Function: compute fundamental matrix from corresponding points
  # YOUR CODE HERE: 
  A = np.zeros((x1.shape[0],9))
  for i in range(x1.shape[0]):
    x = x1[i]
    x_ = x2[i]
    y = y1[i]
    y_ = y2[i]
    A[i,:] = np.array([x*x_ , x*y_ , x , y*x_ , y*y_ , y , x_ , y_ , 1])

  _ , _ , V = np.linalg.svd(A)
  # print(V.shape)
  f = V.T[:,8]
  F = np.reshape(f, (3,3))

  U, S, V = np.linalg.svd(F)
  S = np.array([[S[0],0,0],[0,S[1],0],[0,0,0]])

  F = U @ S @ V

  return F

def denormalize(x1, y1, x2, y2, F):
  # Function: to reverse the normalization process and get the resultant fundamental matrix
  T = normalize(x1, y1, matrix = True)
  T_ = normalize(x2, y2, matrix = True)
  # F_denormalized = np.dot(np.transpose(T_) , np.dot(F, T))
  F_denormalized = T.T @ F @ T_

  return F_denormalized

def error(x1, y1, x2, y2, F):
  a1 = np.hstack((x1, y1, np.ones((x1.shape[0], 1))))
  a2 = np.hstack((x2, y2, np.ones((x2.shape[0], 1))))

  error = 0

  for i in range(a1.shape[0]):
    p1 = a1[i,:].reshape((1,3))
    p2 = a2[i,:].reshape((1,3))

    n1 = F @ p2.T
    d1 = np.abs(p1 @ n1)/np.sqrt(n1[0]**2 + n1[1]**2)

    n2 = F.T @ p1.T
    d2 = np.abs(p2 @ n2)/np.sqrt(n2[0]**2 + n2[1]**2)

    error = error + (d1 + d2)

  error = error/a1.shape[0]
  return error[0,0]

def getBestMatches(indices,x1,y1,x2,y2):
  # Function to filter out the best matches
  a1 = np.hstack((x1, y1, np.ones((x1.shape[0], 1))))
  a2 = np.hstack((x2, y2, np.ones((x2.shape[0], 1))))

  a1 = a1[indices]
  a2 = a2[indices]

  return a1, a2

def drawLines(x1, y1, x2, y2, indices, img1, img2, F, title = '', name = None):
  # Function to visualize the epipolar lines

  rand = np.random.choice(indices.shape[0], 8, replace=False)
  
  # indices = indices[rand]

  coord1, coord2 = getBestMatches(indices, x1, y1, x2, y2)

  ep_lines1 = F @ coord2.T
  ep_lines2 = F.T @ coord1.T

  img = np.zeros((img1.shape[0], img1.shape[1]*2, 3)).astype(np.uint8)
  img[:,0:img1.shape[1],:] = img1
  img[:,img1.shape[1]:img1.shape[1]*2,:] = img2
#   print(np.max(img))
#   plt.imshow(img1)
#   plt.show()

  plt.rcParams["figure.figsize"] = [30, 20]
  fig, ax = plt.subplots()
  im = ax.imshow(img, extent=[0, img.shape[1], 0, img.shape[0]])

  for i in range(coord1.shape[0]):
    x1 = np.linspace(0,img1.shape[1],5)
    y1 = (-ep_lines1[0,i]*x1 - ep_lines1[2,i])/(ep_lines1[1,i])
    y1 = img1.shape[0] - y1
    plt.plot(x1, y1, '-g')
    plt.scatter(coord1[i,0],img1.shape[0]-coord1[i,1],color='r')

    x2 = np.linspace(0,img1.shape[1],5)
    y2 = (-ep_lines2[0,i]*x1 - ep_lines2[2,i])/(ep_lines2[1,i])
    y2 = img1.shape[0] - y2
    plt.plot(x2 + img1.shape[1], y2, '-g')
    plt.scatter(coord2[i,0] + img1.shape[1] ,img1.shape[0]-coord2[i,1],color='r')

  plt.xlim([0, img1.shape[1]*2])
  plt.ylim([0, img1.shape[0]])
  plt.title(title)
  plt.show()
  # plt.savefig("results/" + name + "/" + title + ".png")

def get_essential_matrix(F, K1, K2):
  E = K1.T @ F @ K2
  return E

def get_rot_trans(E):
  # Compute SVD of the essential matrix
  U, S, Vt = np.linalg.svd(E)

  # Ensure that the singular values are positive
  if np.linalg.det(U) < 0:
      U *= -1
  if np.linalg.det(Vt) < 0:
      Vt *= -1

  # Create the skew-symmetric matrix W
  W = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]])

  # Compute the two possible rotation matrices
  R1 = U @ W @ Vt
  R2 = U @ W.T @ Vt

  # Compute the two possible translation vectors
  t1 = U[:, 2]
  t2 = -U[:, 2]

  return [R1, R2], [t1, t2]

def in_front_of_both_cameras(first_points, second_points, rot, trans):
    # check if the point correspondences are in front of both images
    rot_inv = rot
    for first, second in zip(first_points, second_points):
        first_z = np.dot(rot[0, :] - second[0]*rot[2, :], trans) / np.dot(rot[0, :] - second[0]*rot[2, :], second)
        first_3d_point = np.array([first[0] * first_z, second[0] * first_z, first_z])
        second_3d_point = np.dot(rot.T, first_3d_point) - np.dot(rot.T, trans)

        if first_3d_point[2] < 0 or second_3d_point[2] < 0:
            return False

    return True

def process(dataset, scale = 0.5):

  for name in dataset:
    # name = 'chess'

    print(f"\n Case => {name} \n")

    # Load images
    img1 = cv2.imread("data/" + name + "/im0.png")
    img2 = cv2.imread("data/" + name + "/im1.png")

    calib = {'artroom':np.array([[1733.74, 0, 792.27],[0, 1733.74, 541.89],[0, 0, 1]]),
            'chess':np.array([[1758.23,0,829.15],[0,1758.23,552.78],[0, 0, 1]]),
            'ladder':np.array([[1734.16,0,333.49],[0,1734.16,958.05],[0, 0, 1]])}

    baseline = {'artroom':536.62, 'chess':97.99, 'ladder':228.38}

    ndisp = {'artroom':170, 'chess':220, 'ladder':110}

    K = K1 = K2 = calib[name]

    # img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)
    # img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2RGB)

    num_pts = 1000

    # Convert images to grayscale
    gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

    # Initialize SIFT detector
    sift = cv2.xfeatures2d.SIFT_create(num_pts)

    # Find keypoints and descriptors for both images
    kp1, desc1 = sift.detectAndCompute(gray1, None)
    kp2, desc2 = sift.detectAndCompute(gray2, None)

    # Initialize FLANN-based matcher
    bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=False)

    # Match descriptors
    matches = bf.knnMatch(desc1, desc2, k=2)

    # Apply ratio test to select only the good matches
    good_matches = []
    for m, n in matches:
        if m.distance < 0.6 * n.distance:
            good_matches.append(m)

    # Extract x and y coordinates of matching keypoints
    pts1 = []
    pts2 = []

    for match in good_matches:
        pts1.append(kp1[match.queryIdx].pt)
        pts2.append(kp2[match.trainIdx].pt)

    # Computing the Fundamental matrix F
    pts1 = np.array(pts1)
    pts2 = np.array(pts2)

    x1 = pts1[:,0].reshape((len(good_matches),1))
    y1 = pts1[:,1].reshape((len(good_matches),1))
    x2 = pts2[:,0].reshape((len(good_matches),1))
    y2 = pts2[:,1].reshape((len(good_matches),1))

    F, indices = ransacF(x1, y1, x2, y2)
    print(f"\nThe estimated fundamental matrix F is as follows: \n{F}\n")

    # Visualizing Epipolar lines
    drawLines(x1, y1, x2, y2, indices, cv2.cvtColor(img1, cv2.COLOR_BGR2RGB), cv2.cvtColor(img2, cv2.COLOR_BGR2RGB), F, title= "before_rectification", name = name)

    # Getting the essential matrix from the fundamental matrix
    E = get_essential_matrix(F, K1, K2)
    print(f"\nThe estimated essential matrix E is as follows: \n{E}\n")

    R_list, T_list = get_rot_trans(E) 

    K_inv = np.linalg.inv(K)
    homo_norm_pts1 = np.dot(K_inv, np.hstack((pts1, np.ones((pts1.shape[0],1)))).T).T
    homo_norm_pts2 = np.dot(K_inv, np.hstack((pts2, np.ones((pts2.shape[0],1)))).T).T

    for R in R_list:
        for t in T_list:
            flag = in_front_of_both_cameras(homo_norm_pts1, homo_norm_pts2, R, t)
            if flag:
                print(f"\nRotation matrix from Essential matrix decomposition : \n{R}\n")
                print(f"\nTranslation vector from Essential matrix decomposition : \n{t}\n")
                break

    # Computing the rectification homographies
    _, H1, H2 = cv2.stereoRectifyUncalibrated(pts1, pts2, F.T, imgSize=img1.shape[:2])

    print(f"\nHomography matrices for rectification are :\n{H1}\n\n{H2}\n")

    # Rectify the images
    img1_rect = cv2.warpPerspective(img1, H1, (img1.shape[1], img1.shape[0]))
    img2_rect = cv2.warpPerspective(img2, H2, (img2.shape[1], img2.shape[0]))

    trans_pts1 = np.hstack((pts1, np.ones((pts1.shape[0],1))))
    trans_pts2 = np.hstack((pts2, np.ones((pts2.shape[0],1))))

    trans_pts1 = (H1 @ trans_pts1.T).T
    trans_pts1[:,0] = trans_pts1[:,0]/trans_pts1[:,2]
    trans_pts1[:,1] = trans_pts1[:,1]/trans_pts1[:,2]
    trans_pts1[:,2] = trans_pts1[:,2]/trans_pts1[:,2]

    trans_pts2 = (H2 @ trans_pts2.T).T
    trans_pts2[:,0] = trans_pts2[:,0]/trans_pts2[:,2]
    trans_pts2[:,1] = trans_pts2[:,1]/trans_pts2[:,2]
    trans_pts2[:,2] = trans_pts1[:,2]/trans_pts2[:,2]

    # Modify the Fundamental matrix to account for the rectification
    F_trans = np.linalg.inv(H1.T) @ F @ np.linalg.inv(H2)

    trans_x1 = trans_pts1[:,0].reshape((len(good_matches),1))
    trans_y1 = trans_pts1[:,1].reshape((len(good_matches),1))
    trans_x2 = trans_pts2[:,0].reshape((len(good_matches),1))
    trans_y2 = trans_pts2[:,1].reshape((len(good_matches),1))

    # Visualizing the rectified image
    drawLines(trans_x1, trans_y1, trans_x2, trans_y2, indices, cv2.cvtColor(img1_rect, cv2.COLOR_BGR2RGB), cv2.cvtColor(img2_rect, cv2.COLOR_BGR2RGB), F_trans, title= "after_rectification", name = name)

    # Rescaling the rectified images for faster disparity computation
    scale = scale

    img_left = cv2.resize(img1_rect, (int(img1_rect.shape[1] * scale), int(img1_rect.shape[0] * scale)))
    img_right = cv2.resize(img2_rect, (int(img2_rect.shape[1] * scale), int(img2_rect.shape[0] * scale)))

    img_left = cv2.cvtColor(img_left, cv2.COLOR_BGR2GRAY)
    img_right = cv2.cvtColor(img_right, cv2.COLOR_BGR2GRAY)

    # Computing disparity map
    window_size = 11 #window size for search

    left_array, right_array = img_left, img_right
    left_array = left_array.astype(int)
    right_array = right_array.astype(int)

    h, w = left_array.shape
    disparity_map = np.zeros((h, w))

    x_new = w - (2 * window_size)
    for y in tqdm(range(window_size, h-window_size)):
        left_window_array = []
        right_window_array = []
        for x in range(window_size, w-window_size):
            left_window = left_array[y:y + window_size,
                                    x:x + window_size]
            left_window_array.append(left_window.flatten())

            right_window = right_array[y:y + window_size,
                                    x:x + window_size]
            right_window_array.append(right_window.flatten())

        left_window_array = np.array(left_window_array)
        left_window_array = np.repeat(left_window_array[:, :, np.newaxis], x_new, axis=2)

        right_window_array = np.array(right_window_array)
        right_window_array = np.repeat(right_window_array[:, :, np.newaxis], x_new, axis=2)
        right_window_array = right_window_array.T

        abs_diff = np.abs(left_window_array - right_window_array)
        sum_abs_diff = np.sum(abs_diff, axis = 1)
        idx = np.argmin(sum_abs_diff, axis = 0)
        disparity = np.abs(idx - np.linspace(0, x_new, x_new, dtype=int)).reshape(1, x_new)
        disparity_map[y, 0:x_new] = disparity

    # Rescaling the disparity values to accomodate for upsizing
    disparity_map = disparity_map / scale

    # Normalize the disparity map to the range [0, 255] for displaying
    norm_disparity_map = (disparity_map / np.max(disparity_map)) * 255
    norm_disparity_map = norm_disparity_map.astype(np.uint8)

    # Upsizing the disparity matrix to original size
    disparity_map = cv2.resize(disparity_map, img1_rect.shape[:2][::-1], interpolation=cv2.INTER_NEAREST)
    norm_disparity_map = cv2.resize(norm_disparity_map, img1_rect.shape[:2][::-1], interpolation=cv2.INTER_NEAREST)

    cv2.imshow("Disparity map gray", norm_disparity_map)
    cv2.imwrite("results/" + name + "/disp_map_gray.png", norm_disparity_map)

    norm_disparity_map_color = cv2.applyColorMap(norm_disparity_map, cv2.COLORMAP_INFERNO)

    cv2.imshow("Disparity map color", norm_disparity_map_color)
    cv2.imwrite("results/" + name + "/disp_map_color.png", norm_disparity_map_color)

    # Computing the depth map from the disparity map
    depth = (baseline[name] * K[0,0]) / (disparity_map + 1e-10) #Adding buffer to avoid divide by 0 error
    depth[depth > 100000] = 100000

    depth_map = np.uint8(depth * 255 / np.max(depth))

    cv2.imshow("Depth map gray", depth_map)
    cv2.imwrite("results/" + name + "/depth_map_gray.png", depth_map)

    depth_map_color = cv2.applyColorMap(depth_map, cv2.COLORMAP_INFERNO)

    cv2.imshow("Depth map color", depth_map_color)
    cv2.imwrite("results/" + name + "/depth_map_color.png", depth_map_color)

    cv2.waitKey(0)

    cv2.destroyAllWindows()

if __name__ == "__main__":
  process(['artroom', 'chess', 'ladder'])

