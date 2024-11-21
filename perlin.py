import cv2
import numpy as np

def apply_filter1(frame):
    # Apply 50% opacity black overlay
    frame_blended = (frame * 0.5).astype(np.uint8)
    # Convert to grayscale
    gray = cv2.cvtColor(frame_blended, cv2.COLOR_BGR2GRAY)
    # Apply threshold to create binary image
    _, binary = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
    # Create an image with red channel zero, green and blue channels being the binary image
    zeros = np.zeros_like(binary)
    filtered = cv2.merge([zeros, binary, binary])  # R=0, G&B=binary
    return filtered

def apply_filter2(frame):
    # Apply 50% opacity black overlay
    frame_blended = (frame * 0.5).astype(np.uint8)
    # Convert to grayscale
    gray = cv2.cvtColor(frame_blended, cv2.COLOR_BGR2GRAY)
    # Quantize grayscale image to 3 levels: 0, 127, 255
    quantized = np.zeros_like(gray)
    quantized[gray < 85] = 0
    quantized[(gray >= 85) & (gray < 170)] = 127
    quantized[gray >= 170] = 255
    # Create an image with only the red channel
    zeros = np.zeros_like(quantized)
    filtered = cv2.merge([quantized, zeros, zeros])  # R=quantized, G=B=0
    return filtered

def main():
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("Cannot open webcam")
        return

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Cannot read frame")
            break

        # Resize frame if necessary
        # frame = cv2.resize(frame, (640, 480))

        # Apply filters
        filtered1 = apply_filter1(frame)
        filtered2 = apply_filter2(frame)

        # Stack images side by side
        combined = cv2.hconcat([filtered1, filtered2])

        # Display the resulting frame
        cv2.imshow('Filtered Webcam Feed', combined)

        # Break the loop on 'q' key press
        if cv2.waitKey(1) == ord('q'):
            break

    # Release the capture and close windows
    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
