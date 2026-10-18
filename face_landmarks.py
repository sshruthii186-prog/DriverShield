import cv2
import mediapipe as mp
import time

# Load the Face Landmarker model
BaseOptions = mp.tasks.BaseOptions
FaceLandmarker = mp.tasks.vision.FaceLandmarker
FaceLandmarkerOptions = mp.tasks.vision.FaceLandmarkerOptions
RunningMode = mp.tasks.vision.RunningMode

options = FaceLandmarkerOptions(
    base_options=BaseOptions(
        model_asset_path="face_landmarker.task"
    ),
    running_mode=RunningMode.VIDEO,
    num_faces=1
)

camera = cv2.VideoCapture(0)

with FaceLandmarker.create_from_options(options) as landmarker:

    start_time = time.time()
    frame_number = 0

    while camera.isOpened():

        success, frame = camera.read()

        if not success:
            print("Cannot read webcam.")
            break

        # Mirror the webcam image
        frame = cv2.flip(frame, 1)

        # Convert the image to RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Convert to MediaPipe image format
        mp_image = mp.Image(
            image_format=mp.ImageFormat.SRGB,
            data=rgb_frame
        )

        # Calculate timestamp for video processing
        timestamp_ms = int((time.time() - start_time) * 1000)

        # Ensure timestamps increase
        frame_number += 1
        timestamp_ms = max(timestamp_ms, frame_number)

        # Detect facial landmarks
        results = landmarker.detect_for_video(
            mp_image, timestamp_ms
        )

        # Draw landmarks
        if results.face_landmarks:
            for face in results.face_landmarks:
                height, width, _ = frame.shape

                for point in face:
                    x = int(point.x * width)
                    y = int(point.y * height)

                    cv2.circle(
                        frame,
                        (x, y),
                        1,
                        (0, 255, 0),
                        -1
                    )

            cv2.putText(
                frame,
                "Face Detected",
                (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 255, 0),
                2
            )

        else:
            cv2.putText(
                frame,
                "No Face Detected",
                (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 0, 255),
                2
            )

        cv2.imshow("DriverShield - Face Detection", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

camera.release()
cv2.destroyAllWindows()