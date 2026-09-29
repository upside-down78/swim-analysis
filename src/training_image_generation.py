import cv2
import matplotlib as plt
import numpy as np


# Reading testing video
curr_vid = "dan-smith-2"
frame_skip = 8     
max_frames = frame_skip*100
cap = cv2.VideoCapture(f"data/raw/{curr_vid}.mp4")

# Gets video fps to calculate timestamps
frame_num = 0
cap_fps = cap.get(cv2.CAP_PROP_FPS) 
# Gets dimensions of the video
cap_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
cap_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

alive = True

win_name = "Video Window"
cv2.namedWindow(win_name, cv2.WINDOW_NORMAL)

while (alive):
  # Reading frame from video
  has_frame, bgr_frame = cap.read()
  frame_num += 1
  frame_timestamp_ms = int(frame_num*1000/cap_fps)

  if (not has_frame) or (frame_num > max_frames):
        break

  key = cv2.waitKey(1)
  print(f"Frame {frame_num:04d}")
  if frame_num % frame_skip == 0:
    cv2.imwrite(f"training_images/{curr_vid}/frame_{frame_num:04d}.jpg", bgr_frame)
    cv2.imshow(win_name, bgr_frame)
    key = cv2.waitKey(0)
    print("Frame saved.")
    
    if key == ord("Q") or key == ord("q") or key == 27:
      alive = False
    

#out.release()
cap.release()
cv2.destroyAllWindows()
