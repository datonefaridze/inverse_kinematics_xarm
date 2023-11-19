from utils import VideoCapture
import time
import cv2

cap = VideoCapture(2)
idx=0
while True:
  time.sleep(4)   # simulate time between events
  frame = cap.read()
  cv2.imshow("frame", frame)
  print('idx: ', idx)
  idx+=1
  if chr(cv2.waitKey(1)&255) == 'q':
    break