from ultralytics import YOLO

# Load a model
model = YOLO("YOLO11n.pt")  # pretrained YOLO11n model
# Run batched inference on a list of images
results = model(["image1.jpg", "image2.jpg","image3.jpg","images4.jpg"])  # return a list of Results objects

# Process results list
for result in results:
    boxes = result.boxes  # Boxes object for bounding box outputs
    masks = result.masks  # Masks object for segmentation masks outputs
    keypoints = result.keypoints  # Keypoints object for pose outputs
    probs = result.probs  # Probs object for classification outputs
    obb = result.obb  # Oriented boxes object for OBB outputs
    result.show()  # display to screen
    print(result)
    result.save(filename="result.jpg")  # save to disk

