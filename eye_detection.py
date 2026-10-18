
import cv2
import mediapipe as mp
import numpy as np
import time
import winsound
import csv
import os
from datetime import datetime

# Configuration
CLOSURE_LIMIT = 2.0
YAWN_LIMIT = 1.5
HEAD_TURN_LIMIT = 2.0

CSV_FILE = "drowsiness_log.csv"

# Create CSV file if it does not exist
if not os.path.exists(CSV_FILE):
    with open(CSV_FILE, "w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(
            ["Timestamp", "Event", "Duration (seconds)"]
        )

# MediaPipe setup
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

# Eye landmarks
LEFT_EYE = [33, 160, 158, 133, 153, 144]
RIGHT_EYE = [362, 385, 387, 263, 373, 380]

# Mouth landmarks
MOUTH_TOP = 13
MOUTH_BOTTOM = 14
MOUTH_LEFT = 61
MOUTH_RIGHT = 291


def eye_aspect_ratio(points):
    vertical_1 = np.linalg.norm(points[1] - points[5])
    vertical_2 = np.linalg.norm(points[2] - points[4])
    horizontal = np.linalg.norm(points[0] - points[3])

    if horizontal == 0:
        return 0

    return (vertical_1 + vertical_2) / (2.0 * horizontal)


def log_event(event, duration):
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    with open(CSV_FILE, "a", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(
            [timestamp, event, round(duration, 2)]
        )


# Open webcam
camera = cv2.VideoCapture(0)

if not camera.isOpened():
    print("Cannot open webcam. Check whether another application is using it.")
    raise SystemExit(1)

start_time = time.time()
frame_number = 0

# Eye tracking variables
eyes_closed_start = None
drowsiness_logged = False
last_sound_time = 0

# Yawn tracking variables
mouth_open_start = None
yawn_logged = False

# Head-turn tracking variables
head_turn_start = None
head_turn_logged = False

try:
    with FaceLandmarker.create_from_options(options) as landmarker:

        while camera.isOpened():
            success, frame = camera.read()

            if not success:
                print("Cannot read webcam.")
                break

            frame = cv2.flip(frame, 1)
            height, width, _ = frame.shape

            rgb_frame = cv2.cvtColor(
                frame, cv2.COLOR_BGR2RGB
            )

            mp_image = mp.Image(
                image_format=mp.ImageFormat.SRGB,
                data=rgb_frame
            )

            frame_number += 1
            timestamp_ms = max(
                int((time.time() - start_time) * 1000),
                frame_number
            )

            results = landmarker.detect_for_video(
                mp_image, timestamp_ms
            )

            current_time = time.time()

            if results.face_landmarks:
                face = results.face_landmarks[0]

                # -----------------------------
                # 1. EYE-BASED DROWSINESS
                # -----------------------------
                left_eye = np.array([
                    [face[i].x * width, face[i].y * height]
                    for i in LEFT_EYE
                ])

                right_eye = np.array([
                    [face[i].x * width, face[i].y * height]
                    for i in RIGHT_EYE
                ])

                left_ratio = eye_aspect_ratio(left_eye)
                right_ratio = eye_aspect_ratio(right_eye)

                eye_ratio = (left_ratio + right_ratio) / 2

                if eye_ratio < 0.20:
                    if eyes_closed_start is None:
                        eyes_closed_start = current_time

                    closed_duration = (
                        current_time - eyes_closed_start
                    )

                    if closed_duration >= CLOSURE_LIMIT:
                        eye_status = "DROWSINESS ALERT!"
                        eye_color = (0, 0, 255)

                        if not drowsiness_logged:
                            log_event(
                                "Drowsiness Alert",
                                closed_duration
                            )
                            drowsiness_logged = True

                        if current_time - last_sound_time >= 1.0:
                            winsound.Beep(1000, 300)
                            last_sound_time = time.time()

                    else:
                        eye_status = "EYES CLOSED"
                        eye_color = (0, 165, 255)

                else:
                    eye_status = "EYES OPEN"
                    eye_color = (0, 255, 0)

                    eyes_closed_start = None
                    drowsiness_logged = False

                # -----------------------------
                # 2. POSSIBLE YAWNING DETECTION
                # -----------------------------
                mouth_top = np.array([
                    face[MOUTH_TOP].x * width,
                    face[MOUTH_TOP].y * height
                ])

                mouth_bottom = np.array([
                    face[MOUTH_BOTTOM].x * width,
                    face[MOUTH_BOTTOM].y * height
                ])

                mouth_left = np.array([
                    face[MOUTH_LEFT].x * width,
                    face[MOUTH_LEFT].y * height
                ])

                mouth_right = np.array([
                    face[MOUTH_RIGHT].x * width,
                    face[MOUTH_RIGHT].y * height
                ])

                mouth_height = np.linalg.norm(
                    mouth_top - mouth_bottom
                )

                mouth_width = np.linalg.norm(
                    mouth_left - mouth_right
                )

                mouth_ratio = mouth_height / max(mouth_width, 1)

                if mouth_ratio > 0.35:
                    if mouth_open_start is None:
                        mouth_open_start = current_time

                    mouth_duration = (
                        current_time - mouth_open_start
                    )

                    if mouth_duration >= YAWN_LIMIT:
                        mouth_status = "POSSIBLE YAWN DETECTED"
                        mouth_color = (0, 0, 255)

                        if not yawn_logged:
                            log_event(
                                "Possible Yawn",
                                mouth_duration
                            )
                            yawn_logged = True

                    else:
                        mouth_status = "MOUTH OPEN"
                        mouth_color = (0, 165, 255)

                else:
                    mouth_status = "MOUTH NORMAL"
                    mouth_color = (0, 255, 0)

                    mouth_open_start = None
                    yawn_logged = False

                # -----------------------------
                # 3. HEAD-TURN DETECTION
                # -----------------------------
                nose = face[1]
                left_side = face[234]
                right_side = face[454]

                face_width = abs(
                    right_side.x - left_side.x
                )

                if face_width > 0:
                    relative_position = (
                        (nose.x - left_side.x) / face_width
                    )

                    if (
                        relative_position < 0.35
                        or relative_position > 0.65
                    ):
                        head_status = "POSSIBLE HEAD TURN"
                        head_color = (0, 0, 255)

                        if head_turn_start is None:
                            head_turn_start = current_time

                        head_duration = (
                            current_time - head_turn_start
                        )

                        if head_duration >= HEAD_TURN_LIMIT:
                            if not head_turn_logged:
                                log_event(
                                    "Possible Head Turn",
                                    head_duration
                                )
                                head_turn_logged = True

                    else:
                        head_status = "HEAD POSITION NORMAL"
                        head_color = (0, 255, 0)

                        head_turn_start = None
                        head_turn_logged = False

                else:
                    head_status = "HEAD POSITION UNKNOWN"
                    head_color = (0, 165, 255)

                    head_turn_start = None
                    head_turn_logged = False

                # -----------------------------
                # 4. DISPLAY STATUS
                # -----------------------------
                cv2.putText(
                    frame,
                    eye_status,
                    (20, 35),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    eye_color,
                    2
                )

                cv2.putText(
                    frame,
                    f"Eye Ratio: {eye_ratio:.3f}",
                    (20, 65),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.55,
                    eye_color,
                    2
                )

                cv2.putText(
                    frame,
                    mouth_status,
                    (20, 100),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    mouth_color,
                    2
                )

                cv2.putText(
                    frame,
                    f"Mouth Ratio: {mouth_ratio:.3f}",
                    (20, 130),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.55,
                    mouth_color,
                    2
                )

                cv2.putText(
                    frame,
                    head_status,
                    (20, 165),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    head_color,
                    2
                )

                # Draw eye landmarks
                for point in np.vstack((left_eye, right_eye)):
                    cv2.circle(
                        frame,
                        (int(point[0]), int(point[1])),
                        3,
                        eye_color,
                        -1
                    )

                # Draw mouth landmarks
                for point in [
                    mouth_top,
                    mouth_bottom,
                    mouth_left,
                    mouth_right
                ]:
                    cv2.circle(
                        frame,
                        (int(point[0]), int(point[1])),
                        4,
                        mouth_color,
                        -1
                    )

            else:
                eyes_closed_start = None
                drowsiness_logged = False

                mouth_open_start = None
                yawn_logged = False

                head_turn_start = None
                head_turn_logged = False

                cv2.putText(
                    frame,
                    "NO FACE DETECTED",
                    (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (0, 0, 255),
                    2
                )

            cv2.imshow(
                "DriverShield - Driver Monitoring",
                frame
            )

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

finally:
    camera.release()
    cv2.destroyAllWindows()

print("DriverShield detection stopped.")

