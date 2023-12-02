import cv2, queue, threading, time
import time


class VideoCapture:

  def __init__(self, name):
    self.cap = cv2.VideoCapture(name)
    self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 128)
    self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 128)

    self.q = queue.Queue()
    t = threading.Thread(target=self._reader)
    t.daemon = True
    t.start()

  # read frames as soon as they are available, keeping only most recent one
  def _reader(self):
    while True:
      ret, frame = self.cap.read()
      if not ret:
        break
      if not self.q.empty():
        try:
          self.q.get_nowait()   # discard previous (unprocessed) frame
        except queue.Empty:
          pass
      self.q.put(frame)

  def read(self):
    return self.q.get()

# Start the video capture
cap = VideoCapture(2)  # 0 is usually the default camera

# Set the desired resolution
# cap.set(cv2.CAP_PROP_FRAME_WIDTH, 256)
# cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 256)

a = 0
start = time.time()
while True:
    frame = cap.read()  # Read a frame
    ret = True
    total = time.time()-start
    if total >=1 :
        break
    a += 1

    if not ret:
        break  # Exit the loop if no frame is captured
    print("frame: ", frame.shape)
    # The 'frame' should now be 256x256, but this depends on camera support
    # Display the frame
    # cv2.imshow('Frame', frame)

    # Wait for a key press and break the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the capture and close any open windows
print(a)
cap.release()
cv2.destroyAllWindows()
