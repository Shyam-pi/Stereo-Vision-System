# Stere Vision System

**Note**: This project was done as a part of the course ENPM673 - Perception for Autonomous Robots in Spring 2023 at the University of Maryland, College Park.

## Project Overview

This project focuses on implementing a stereo vision system to estimate the depth of objects in a scene using two calibrated cameras. The pipeline involves calibration, fundamental matrix computation, estimation of camera pose, rectification, correspondence, and depth map computation.

## Pipeline and Mathematics

### Calibration

#### Steps:
1. **Feature Detection and Matching:**
   - Detect features in the stereo images using the SIFT feature detector.
   - Match the detected features between the images using the ratio test.

2. **Fundamental Matrix Computation using RANSAC:**
   - Compute the fundamental matrix \( F \) using RANSAC.
   - Normalize the matching points and formulate the \( A \) matrix for fundamental matrix computation.
   - Compute the denormalized fundamental matrix \( F \).

3. **Estimation of Camera Pose:**
   - Compute the essential matrix \( E \) from the fundamental matrix.
   - Estimate the camera rotation matrix \( R \) and translation vector \( C \) from \( E \) using the algorithm.
   - Select the correct rotation and translation pair satisfying the chirality condition.

### Rectification

#### Steps:
1. **Compute Perspective Transformations:**
   - Compute perspective transformations \( H_1 \) and \( H_2 \) for rectification using stereoRectifyUncalibrated.

2. **Warp Images:**
   - Warp the stereo images using \( H_1 \) and \( H_2 \) to obtain rectified images.

3. **Update Fundamental Matrix:**
   - Update the fundamental matrix \( F \) to account for the perspective transformations.

### Correspondence

#### Steps:
1. **Row-wise Search and Matching:**
   - Perform a row-wise search for corresponding points in the rectified images.
   - Compute the disparity map using Sum of Squared Differences (SSD) matching.

### Depth Map Computation

#### Steps:
1. **Compute Depth:**
   - Compute the depth map using the baseline length, focal length, and pixel-wise disparity.

### Results

The project results include visualizations of epipolar lines, disparity maps, and depth maps for various datasets such as artroom, chess, and ladder. Artroom results are shown here:

Required matrices estimated : 

![image](https://github.com/Shyam-pi/Stereo-Vision-System/assets/57116285/a2214b81-be72-449d-a979-223810efd5e9)

Epipolar lines before rectification :

![image](https://github.com/Shyam-pi/Stereo-Vision-System/assets/57116285/00940086-17c7-4e24-b735-1fd56a81f3ae)

Epipolar lines after rectification :

![image](https://github.com/Shyam-pi/Stereo-Vision-System/assets/57116285/822eb004-fd5a-44e3-bae0-200e72e560a7)

Disparity Map :

![image](https://github.com/Shyam-pi/Stereo-Vision-System/assets/57116285/22c760e0-a4be-40cf-bff4-db027ee30342)

Depth Map :

![image](https://github.com/Shyam-pi/Stereo-Vision-System/assets/57116285/de8c1350-af17-4d50-8f38-a4b674d08af4)


## Contributions

Contributions to this repository are welcome. If you find any issues or have suggestions for improvements, please open an issue or create a pull request.
