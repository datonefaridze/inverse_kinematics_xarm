import cv2
import datetime
import time
from utils import VideoCapture

cap = VideoCapture(0)


def read_times(cap, duration):
    duration = 0.2  # Duration in seconds
    start_time = time.time()

    while True:
        frame = cap.read()
        elapsed_time = time.time() - start_time
        if elapsed_time >= duration:
            break  # Break the loop after 0.2 seconds

time.sleep(10)
while True:
    frame = cap.read()


    timestamp = datetime.datetime.now()
    cv2.putText(frame, timestamp.strftime("%Y-%m-%d %H:%M:%S.%f"), (10, 30), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    cv2.imshow('Frame', frame)
    # time.sleep(5)
    # read_times(cap, 0.2)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
