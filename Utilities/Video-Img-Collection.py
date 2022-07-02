import cv2

print('Splitting Video Into Frames...')

vidcap = cv2.VideoCapture('Monaco-Highlights.mp4')
success,image = vidcap.read()
count = 0

while success:
    # Write every 10th Frame to save space
    if count % 10 == 0:
        cv2.imwrite("//Users//jamesburkinshaw//Desktop//Masters//Machine Learning - May 2022//Assignments//F1-TF//Monaco Frames//frame%d.png" % count, image)       
    
    success,image = vidcap.read()
    count += 1
  
print('Done')
