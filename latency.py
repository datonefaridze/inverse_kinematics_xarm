import cv2
import datetime
import time

cap = cv2.VideoCapture(2)
idx=0
while cap.isOpened():
    idx+=1
    ret, frame = cap.read()
    print('real time now:', idx,  datetime.datetime.now())
    if not ret:
        break
    cv2.putText(frame, str(datetime.datetime.now()), (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.imshow('Frame', frame)
    time.sleep(1)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
