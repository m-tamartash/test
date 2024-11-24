import cv2
import concurrent.futures
from ultralytics import YOLO

# Load the YOLO model on CPU
model = YOLO("CarMotorLicense.onnx", task='detect')

# List of RTSP streams
camera_streams = [
    r"rtsp://192.168.1.203:5554/Demo.mp4",
    # r"rtsp://192.168.1.203:5554/Demo.mp4",
]


# Processing each camera stream
def process_camera_stream(rtsp_url, output_name, window_name, model):
    # Open the video stream
    cap = cv2.VideoCapture(rtsp_url)

    # Get video details
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)

    # Define the codec and create VideoWriter to save the output video for each camera
    out = cv2.VideoWriter(output_name, cv2.VideoWriter_fourcc(*'mp4v'), fps, (frame_width, frame_height))

    # Define ROI (Region of Interest) - set custom values based on your requirement
    x, y, w, h = 0, 400, 1500, 500

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Define the region of interest (ROI)
        roi = frame[y:y + h, x:x + w]

        # Run YOLO inference on the ROI
        results = model(roi)

        # Draw the results (bounding boxes, etc.) on the ROI
        annotated_roi = results[0].plot()

        # Replace the original ROI with the annotated ROI in the original frame
        frame[y:y + h, x:x + w] = annotated_roi

        # Draw the ROI rectangle on the frame
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 1)

        # Write the annotated frame to the output video
        out.write(frame)

        # Display the video in a unique window
        cv2.imshow(window_name, frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release resources
    cap.release()
    out.release()
    cv2.destroyWindow(window_name)


# Using ThreadPoolExecutor to process multiple RTSP streams concurrently
with concurrent.futures.ThreadPoolExecutor() as executor:
    futures = []
    for idx, rtsp_url in enumerate(camera_streams):
        # Define an output file name and unique window name for each camera stream
        output_name = f"output_camera_{idx + 1}.mp4"
        window_name = f"Camera {idx + 1}"  # Unique window name for each camera

        # Start a new thread for each camera stream
        futures.append(executor.submit(process_camera_stream, rtsp_url, output_name, window_name, model))

    # Wait for all threads to complete
    for future in concurrent.futures.as_completed(futures):
        future.result()  # This will raise exceptions if any occurred in the threads

# Ensure all windows are closed at the end
cv2.destroyAllWindows()
