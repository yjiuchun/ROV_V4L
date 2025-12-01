from ultralytics import YOLO
import time
# Load a model
model = YOLO("yolo11n.pt")  # pretrained YOLO11n model


start_time = time.time()
# Run batched inference on a list of images
results = model(["/home/yjc/Project/rov_ws/dataset/eight_led/image/499.jpg"])  # return a list of Results objects
end_time = time.time()
print(f"Time taken: {end_time - start_time} seconds")
# Process results list
for result in results:
    boxes = result.boxes  # Boxes object for bounding box outputs
    masks = result.masks  # Masks object for segmentation masks outputs
    keypoints = result.keypoints  # Keypoints object for pose outputs
    probs = result.probs  # Probs object for classification outputs
    obb = result.obb  # Oriented boxes object for OBB outputs
    # result.show()  # display to screen
    result.save(filename="result.jpg")  # save to disk