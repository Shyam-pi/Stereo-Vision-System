NOTE: 
1. Reused some codes for fundamental matrix calculation from my previous work in the course CMSC733 - Computer Processing of Pictorial Information. The python script for the project execution is stereo.py, however please note that it might take around 7 minutes to process and give results for each dataset (30 seconds for feature match and F computation, E computation and rectification including plotting. The rest of the time for disparity matrix computation)
2. The results corresponding to each dataset can be found in the 'results' folder.
3. Please install tqdm and opencv-contrib packages using the following terminal commands:
'pip install tqdm' & 'pip install opencv-contrib-python'

Steps:

1. Open the unzipped project folder in VSCode such that the project folder becomes the root folder for the script (Important so that there are no data reading errors)
2. Execute the script 'stereo.py' by the following command in the terminal - 'python3 stereo.py'

Comments:

 1. Upon execution of the script, it goes through each of the datasets in the following order - 'artroom' -> 'chess' -> 'ladder'
 2. For each dataset, it prints all the required matrices in the terminal, followed by two matplotlib pop-ups showing the epipolar lines before and after rectification. You must close these windows for the script to proceed running.
 3. This is followed by computation of the disparity map. There's a bar showing the progress and time left for that to finish.
 4. Upon completion of the disparity map computation, 4 opencv windows pop up showing the gray and color map versions of the disparity map and depth map.
 5. Press any key on your keyboard to close these windows and let the script move to the next dataset.
 