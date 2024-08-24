import torch, cv2, time
import numpy as np
from PIL import Image
from transformers import pipeline

width = 1920
height = 1080
cam_fps = 30
#factor to scale down to run models (improves performance dramtically)
#dropping lower causes significant accuracy loss
scale_down = .2

#Suppress torch cuda autocast deprecation
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

#Load the YOLOv5 model
model = torch.hub.load('ultralytics/yolov5', 'yolov5n')

#load depth anything pipeline
if torch.cuda.is_available():
  print("CUDA FOUND")
  pipe = pipeline(task="depth-estimation", model="depth-anything/Depth-Anything-V2-Small-hf", device="cuda")
else:
  print("NO CUDA FOUND MODEL PERFORMANCE SIGNIFICANTLY DEGRADED")
  pipe = pipeline(task="depth-estimation", model="depth-anything/Depth-Anything-V2-Small-hf")

#pipe = pipeline(task="depth-estimation", model="depth-anything/Depth-Anything-V2-Small-hf", device="cuda")

depth_mapping = True
object_detection = True

#runs image to pipeline and applys result array to a cv2 color map
def depth_processing(pil_image, img):
  depth_result = pipe(pil_image)["depth"]
  depth_array = np.array(depth_result)
  depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth_array, alpha=255.0/depth_array.max()), cv2.COLORMAP_HOT) #use BONE for grayscale
  depth_colormap_resized = cv2.resize(depth_colormap, (img.shape[1], img.shape[0]))

  return depth_colormap_resized

# scale a point up by the factor
def scaleUp(val: tuple):
  temp0 = int(val[0] * (1/scale_down))
  temp1 = int(val[1] * (1/scale_down))
  return (temp0, temp1)

#process pipeline
def process_webcam(mirror=False):
    cam = cv2.VideoCapture(0)
    # Set camera resolution to 1920x1080
    cam.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cam.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

    # Set camera frame rate to fps
    cam.set(cv2.CAP_PROP_FPS, cam_fps)

    # FPS counter variables
    fps = 0
    frame_count = 0
    start_time = time.time()

    while True:
      ret_val, img = cam.read()
      if mirror: 
          img = cv2.flip(img, 1)
      
      #SCALING IMAGE DOWN FOR PERFORMANCE
      img = cv2.resize(img, None, fx= scale_down, fy= scale_down)
      color_converted = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
      pil_image = Image.fromarray(color_converted)


      
      if object_detection:
        results = model(pil_image)
        objects = []
        # Get bounding box coordinates and labels
        for box in results.xyxy[0].cpu().numpy():
            x1, y1, x2, y2, confidence, class_id = box
            center = ( int( (int(x2)+int(x1)) / 2 ), int( (int(y2)+int(y1)) / 2 ) )
            #if person tell to move
            """
            if results.names[int(class_id)] == "person":
              print(f"Person detected at: {center}")
              if center[0] > width/2:
                  print("Turn right")
              else:
                  print("Turn left")
            """
            #if float(confidence) < 0.5:
            #  continue
            object = {
              "label": f'{results.names[int(class_id)]} {confidence:.2f}',
              "center": center,
              "rectangle1":(int(x1), int(y1)),
              "rectangle2": (int(x2), int(y2)),
            }

            objects.append(object)
            #label = f'{results.names[int(class_id)]} {confidence:.2f}'

            # Draw bounding box
            #cv2.rectangle(img, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
            
            #cv2.circle(img, center, 2, (0, 255, 0), 1)

            #cv2.putText(img, label, (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
          

      # Calculate FPS
      frame_count += 1
      elapsed_time = time.time() - start_time
      if elapsed_time > 1:
          fps = frame_count / elapsed_time
          frame_count = 0
          start_time = time.time()

      #run depth processing pipeline
      if depth_mapping:
        img = depth_processing(pil_image, img)
      
      #print(objects)
      img = cv2.resize(img, None, fx= 1/scale_down, fy= 1/scale_down)
      # Display FPS on the frame
      cv2.putText(img, f'FPS: {fps:.2f}', (10, 20), cv2.FONT_HERSHEY_SIMPLEX, .5, (0, 255, 0))

      #draw bounding boxes
      for object in objects:
        cv2.rectangle(img, scaleUp(object["rectangle1"]), scaleUp(object["rectangle2"]), (0, 255, 0), 2)
        
        cv2.circle(img, scaleUp(object["center"]), 2, (0, 255, 0), 1)

        cv2.putText(img, object["label"], scaleUp(object["rectangle1"]), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

      cv2.imshow('Feed', img)

      if cv2.waitKey(1) == 27: 
          break  # esc to quit
    cv2.destroyAllWindows()

if __name__ == '__main__':
    process_webcam(mirror=True)