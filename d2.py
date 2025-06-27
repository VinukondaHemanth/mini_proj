import streamlit as st
import tempfile
import cv2
import os
from ultralytics import YOLO
import numpy as np

st.title("YOLO Detection for Images and Video")

# Load model
model = YOLO("YOLO11n.pt")  # Replace with correct model name

# Upload file
uploaded_file = st.file_uploader("Upload an image or video", type=["jpg", "jpeg", "png", "mp4", "mov"])

if uploaded_file is not None:
    file_type = uploaded_file.type

    # Handle image
    if "image" in file_type:
        st.image(uploaded_file, caption="Uploaded Image", use_column_width=True)
        with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp_img:
            tmp_img.write(uploaded_file.read())
            tmp_img_path = tmp_img.name

        results = model(tmp_img_path)

        for result in results:
            annotated = result.plot()
            st.image(annotated, caption="Detection Result", use_column_width=True)

    # Handle video
    elif "video" in file_type:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp_vid:
            tmp_vid.write(uploaded_file.read())
            tmp_vid_path = tmp_vid.name

        st.video(tmp_vid_path)

        cap = cv2.VideoCapture(tmp_vid_path)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)

        # Save output video
        out_path = "output_video.mp4"
        out = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))

        frame_count = 0
        st.write("Processing video...")

        while cap.isOpened():
            success, frame = cap.read()
            if not success:
                break

            # Run YOLO detection
            results = model(frame)

            for result in results:
                annotated = result.plot()
                out.write(annotated)  # save frame

            frame_count += 1
            if frame_count % 10 == 0:
                st.write(f"Processed {frame_count} frames...")

        cap.release()
        out.release()

        st.video(out_path)
        st.success("Detection Finished ✅")
