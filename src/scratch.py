import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import cv2
import matplotlib as plt
from cap_from_youtube import cap_from_youtube
import yt_dlp

# Reading testing video
cap = cv2.VideoCapture("data/raw/dan-smith.mp4")

# Gets video fps to calculate timestamps
frame_num = 0
cap_fps = cap.get(cv2.CAP_PROP_FPS) 
# Gets dimensions of the video
cap_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
cap_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# Create output video file
out = cv2.VideoWriter("data/processed/output.avi", cv2.VideoWriter_fourcc(*'XVID'), cap_fps, (cap_width, cap_height))
 

alive = True

win_name = "Video Window"
model_path = "models/pose_landmarker_heavy.task" # Specifying mediapipe landmarker model
#cv2.namedWindow(win_name, cv2.WINDOW_NORMAL)


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
      x = right_wrist.x
      y = right_wrist.y
      z = right_wrist.z
      visibility = right_wrist.visibility

      # Adjust coordinates to pixel positions
      px = int(x * cap_width)
      py = int(y * cap_height)

      cv2.circle(result, (px, py), 10, (0, 0, 255), 5)
      
      # print(f"Right Wrist - X: {x:.2f}, Y: {y:.2f}, Z: {z:.2f}, Visibility: {visibility:.2f}")

    out.write(result)
    
    '''
    key = cv2.waitKey(0)
    if key == ord("Q") or key == ord("q") or key == 27:
      alive = False
    '''

out.release()
cap.release()
cv2.destroyAllWindows()
  