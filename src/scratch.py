import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import cv2
import matplotlib as plt
from cap_from_youtube import cap_from_youtube
import yt_dlp
import numpy as np

# Reading testing video
cap = cv2.VideoCapture("data/raw/person-walking.mp4")

# Gets video fps to calculate timestamps
frame_num = 0
cap_fps = cap.get(cv2.CAP_PROP_FPS) 
# Gets dimensions of the video
cap_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
cap_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# Create output video file
#out = cv2.VideoWriter("data/processed/person-walking-output.avi", cv2.VideoWriter_fourcc(*'XVID'), cap_fps, (cap_width, cap_height))
 

alive = True

win_name = "Video Window"
model_path = "models/pose_landmarker_heavy.task" # Specifying mediapipe landmarker model
cv2.namedWindow(win_name, cv2.WINDOW_NORMAL)


# Initializing pose_landmarker specs
BaseOptions = mp.tasks.BaseOptions
PoseLandmarker = mp.tasks.vision.PoseLandmarker
PoseLandmarkerOptions = mp.tasks.vision.PoseLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode

options = PoseLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=model_path),
    running_mode=VisionRunningMode.VIDEO,
    min_tracking_confidence=0.0)

with PoseLandmarker.create_from_options(options) as landmarker:
  while (alive):
    # Reading frame from video
    has_frame, bgr_frame = cap.read()
    frame_num += 1
    frame_timestamp_ms = int(frame_num*1000/cap_fps)

    if not has_frame:
          break

    # Converting from OpenCV's BGR format to standard RGB
    rgb_frame = cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2RGB)

    # Creating mediapipe image object
    mp_image_rgb = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
    # Detecting key body parts
    pose_landmarks_result = landmarker.detect_for_video(mp_image_rgb, frame_timestamp_ms)

    #print("x")
    result = bgr_frame
    if pose_landmarks_result.pose_landmarks:
      # Landmarks for the first detected person
      first_person_landmarks = pose_landmarks_result.pose_landmarks[0]
      
      # Extract right wrist data from index 16
      right_wrist = first_person_landmarks[16]
      
      # Extract normalized coordinates
      right_x = right_wrist.x
      right_y = right_wrist.y
      right_z = right_wrist.z
      right_visibility = right_wrist.visibility


      # Adjust coordinates to pixel positions
      right_px = int(right_x * cap_width)
      right_py = int(right_y * cap_height)

      cv2.circle(result, (right_px, right_py), 10, (0, 0, 255), 5)
      
      print(f"Right Wrist - X: {right_px:.2f}, Y: {right_py:.2f}, Z: {right_z:.2f}, Visibility: {right_visibility:.2f}")

      # Extract left wrist data from index 15
      left_wrist = first_person_landmarks[23]
      
      # Extract normalized coordinates
      left_x = left_wrist.x
      left_y = left_wrist.y
      left_z = left_wrist.z
      left_visibility = left_wrist.visibility


      # Adjust coordinates to pixel positions
      left_px = int(left_x * cap_width)
      left_py = int(left_y * cap_height)

      cv2.circle(result, (left_px, left_py), 10, (0, 255, 0), 5)
      
      print(f"Left Wrist - X: {left_px:.2f}, Y: {left_py:.2f}, Z: {left_z:.2f}, Visibility: {left_visibility:.2f}")

    #out.write(result)
    
    cv2.imshow(win_name, result)
    key = cv2.waitKey(0)
    if key == ord("Q") or key == ord("q") or key == 27:
      alive = False
    

#out.release()
cap.release()
cv2.destroyAllWindows()
