🚗 DriverShield – AI-Based Driver Monitoring System



📌 Project Overview



DriverShield is an AI-based driver monitoring prototype designed to identify possible signs of driver drowsiness and distraction using a webcam.



The system processes facial landmarks to estimate eye closure, mouth opening, and head position. It records detected events in a CSV file and displays them on an interactive dashboard.



🎯 Objectives



Detect prolonged eye closure.

Identify possible yawning.

Estimate possible head turns.

Record detected events with timestamps and durations.

Display event statistics and history through a dashboard.



✨ Features



1. Drowsiness Detection



Uses facial landmarks to estimate eye closure and identify prolonged eye closure.



2. Possible Yawn Detection



Estimates mouth opening and records possible yawning events.



3. Possible Head-Turn Detection



Estimates head position using facial landmarks and records possible head turns.



4. Event Logging



Stores detected events, timestamps, and estimated durations in `drowsiness\_log.csv`.


5. Interactive Dashboard



Displays:



Total recorded events

Drowsiness alerts

Possible yawns

Possible head turns

Event summary chart

Event history table

CSV download option



 🛠️ Technologies Used



Python

OpenCV

MediaPipe

NumPy

Pandas

Streamlit

CSV file handling



📂 Project Structure



text

DriverShield/

├── dashboard.py

├── eye\_detection.py

├── face\_landmarks.py

├── head\_test.py

├── webcam\_test.py

├── face\_landmarker.task

├── drowsiness\_log.csv

└── README.md





The `venv` folder contains the Python virtual environment and is not required to be uploaded to GitHub.



 ⚙️ Installation and Setup



Step 1: Clone the repository



bash

git clone YOUR\_GITHUB\_REPOSITORY\_URL





Replace the placeholder with your actual repository URL.



Step 2: Enter the project folder



bash

cd DriverShield





Step 3: Create and activate a virtual environment



On Windows Git Bash:



bash

python -m venv venv

source venv/Scripts/activate





Step 4: Install dependencies



Install the required packages:



bash

python -m pip install opencv-python mediapipe numpy pandas streamlit





Step 5: Run the monitoring system



bash

python eye\_detection.py




Allow webcam access when prompted. Press `Q` to close the monitoring window.



Step 6: Launch the dashboard



Open another terminal, activate the virtual environment, navigate to the project folder, and run:


bash

python -m streamlit run dashboard.py





Open the local address displayed in the terminal, usually `http://localhost:8501`.



📊 Output



The system records possible drowsiness, yawning, and head-turn events in a CSV file. The dashboard presents event counts, a chart, and event history.



⚠️ Limitations



DriverShield is an educational prototype. Its detections are estimates and may produce false positives or miss actual events. It has not been validated for real-world road safety and must not be relied upon to determine whether someone is safe to drive.



Test the system only in a safe, stationary setting.



🔮 Future Enhancements



Improve head-pose estimation.

Calibrate detection thresholds across different users and lighting conditions.

Add configurable alerts.

Explore emergency notification features.

Evaluate detection performance using a labelled test dataset.



👩‍💻 Project Purpose



This project demonstrates how computer vision and facial landmark analysis can be used to develop a basic driver-monitoring prototype.



