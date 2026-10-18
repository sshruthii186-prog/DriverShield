import cv2

camera = cv2.VideoCapture(0)

if not camera.isOpened():
    print("Error: Could not open webcam.")

else:
    print("Webcam opened successfully!")

    while True:
        success, frame = camera.read()

        if not success:
            print("Could not read video frame.")
            break

        cv2.imshow("DriverShield - Webcam Test", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

camera.release()
cv2.destroyAllWindows()
