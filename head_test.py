import cv2
import mediapipe as mp
import numpy as np
import time

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

start_time = time.time()
frame_number = 0

with FaceLandmarker.create_from_options(options) as landmarker:

    while camera.isOpened():
        success, frame = camera.read()

        if not success:
            break

        frame = cv2.flip(frame, 1)
        height, width, _ = frame.shape

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        image = mp.Image(
            image_format=mp.ImageFormat.SRGB,
            data=rgb
        )

        frame_number += 1
        timestamp_ms = max(
            int((time.time() - start_time) * 1000),
            frame_number
        )

        result = landmarker.detect_for_video(image, timestamp_ms)

        if result.face_landmarks:
            face = result.face_landmarks[0]

            # Nose and face reference points
            nose = face[1]
            left_side = face[234]
            right_side = face[454]

            nose_x = nose.x * width
            left_x = left_side.x * width
            right_x = right_side.x * width

            face_width = abs(right_x - left_x)

            if face_width > 0:
                relative_position = (
                    (nose_x - left_x) / face_width
                )

                if relative_position < 0.35:
                    status = "HEAD TURNED"
                elif relative_position > 0.65:
                    status = "HEAD TURNED"
                else:
                    status = "HEAD POSITION NORMAL"

                cv2.putText(
                    frame,
                    status,
                    (20, 45),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (0, 255, 0) if status == "HEAD POSITION NORMAL"
                    else (0, 0, 255),
                    2
                )

                cv2.circle(
                    frame,
                    (int(nose_x), int(nose.y * height)),
                    5,
                    (255, 0, 0),
                    -1
                )

        else:
            cv2.putText(
                frame,
                "NO FACE DETECTED",
                (20, 45),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 0, 255),
                2
            )

        cv2.imshow("DriverShield - Head Position Test", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

camera.release()
cv2.destroyAllWindows()