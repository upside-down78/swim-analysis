from roboflow import Roboflow
from dotenv import load_dotenv
import os

load_dotenv()
print("Key loaded:", os.environ.get("ROBOFLOW_API_KEY") is not None)
rf = Roboflow(api_key=os.environ["ROBOFLOW_API_KEY"])
project = rf.workspace(os.environ["ROBOFLOW_WORKSPACE_NAME"]).project(os.environ["ROBOFLOW_PROJECT_NAME"])
version = project.version(2)
dataset = version.download("yolov8")