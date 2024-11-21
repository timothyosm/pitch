import cv2
import numpy as np

def main():
    # Open the default webcam
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("Cannot open webcam")
        return

    while True:
        # Read frame from the webcam
        ret, frame = cap.read()

        if not ret:
            print("Failed to grab frame. Exiting...")
            break

        # Resize the frame (optional)
        # frame = cv2.resize(frame, (640, 480))

        # Convert the frame to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Apply Gaussian blur to simulate heat diffusion
        blur = cv2.GaussianBlur(gray, (21, 21), 0)

        # Normalize the pixel values
        normalized = cv2.normalize(blur, None, 0, 255, cv2.NORM_MINMAX)

        # Apply a thermal color map
        thermal = cv2.applyColorMap(normalized, cv2.COLORMAP_INFERNO)

        # Display the thermal image
        cv2.imshow('Thermal Camera Simulation', thermal)

        # Exit on pressing 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release resources
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
