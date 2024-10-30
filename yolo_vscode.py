import streamlit as st
import os
import tempfile
import cv2
import pandas as pd
from ultralytics import YOLO

# Load YOLO model
def load_model(model_path='D:/Rahul/Work/Oct/29-10-2024/VS Code/yolo_object_detection_project - Copy/models/best.pt'):
    model = YOLO(model_path)  # Load with ultralytics
    return model

# Run inference on image
def detect_image(model, image_path, output_dir='D:/Rahul/Work/Oct/29-10-2024/VS Code/yolo_object_detection_project - Copy/output'):
    results = model(image_path)
    os.makedirs(output_dir, exist_ok=True)

    class_names = []
    for idx, result in enumerate(results):
        if hasattr(result, 'boxes'):
            for box in result.boxes:
                class_name = model.names[int(box.cls[0].item())]
                class_names.append(class_name)

        output_path = os.path.join(output_dir, f'result_{idx}.jpg')
        if hasattr(result, 'plot'):
            annotated_image = result.plot()
            cv2.imwrite(output_path, annotated_image)

    result_df = pd.DataFrame(class_names, columns=["Detected Class"])
    return result_df

# Run inference on video with real-time display
def detect_video_frame_by_frame(model, video_path, stframe, output_dir='D:/Rahul/Work/Oct/29-10-2024/VS Code/yolo_object_detection_project - Copy/output'):
    cap = cv2.VideoCapture(video_path)
    os.makedirs(output_dir, exist_ok=True)

    output_video_path = os.path.join(output_dir, 'annotated_video.mp4')
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_video_path, fourcc, cap.get(cv2.CAP_PROP_FPS),
                          (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))))

    class_names = []
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        results = model(frame)
        frame_class_names = []
        if hasattr(results[0], 'boxes'):
            for box in results[0].boxes:
                class_name = model.names[int(box.cls[0].item())]
                frame_class_names.append(class_name)

        annotated_frame = results[0].plot()
        out.write(annotated_frame)

        # Send annotated frame to Streamlit for real-time display
        stframe.image(annotated_frame, channels="BGR")

        class_names.extend(frame_class_names)

    cap.release()
    out.release()

    result_df = pd.DataFrame(class_names, columns=["Detected Class"]).drop_duplicates()
    return output_video_path, result_df

# Initialize the YOLO model
model = load_model()

# Streamlit interface
st.title("YOLO Object Detection")
st.write("Upload an image or video to detect objects.")

# Upload an image or video
uploaded_file = st.file_uploader("Choose an image or video...", type=["jpg", "jpeg", "png", "mp4"])

if uploaded_file:
    if uploaded_file.type.startswith("image"):
        # Handle image upload and detection
        file_path = os.path.join("data", uploaded_file.name)
        with open(file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())

        # Run detection on the image
        results_df = detect_image(model, file_path)

        # Display results for the image
        st.image(file_path, caption="Uploaded Image", use_column_width=True)
        st.write("Detection Results:")
        st.write(results_df)

    elif uploaded_file.type == "video/mp4":
        # Handle video upload and detection
        tfile = tempfile.NamedTemporaryFile(delete=False)
        tfile.write(uploaded_file.read())
        video_path = tfile.name

        # Run detection on the video frame-by-frame
        stframe = st.empty()
        output_video_path, results_df = detect_video_frame_by_frame(model, video_path, stframe)

        # Display final annotated video
        st.video(output_video_path)
        st.write("Detection Results:")
        st.write(results_df)
