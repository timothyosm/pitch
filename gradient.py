import cv2
import numpy as np
import mediapipe as mp

def create_radial_gradient(size, center):
    """
    Creates a radial gradient image of given size centered at 'center'.
    """
    y_indices, x_indices = np.indices((size[0], size[1]))
    distance = np.sqrt((x_indices - center[0])**2 + (y_indices - center[1])**2)
    max_distance = np.sqrt((size[0] / 2) ** 2 + (size[1] / 2) ** 2)
    gradient = (1 - distance / max_distance) * 255
    gradient = np.clip(gradient, 0, 255).astype(np.uint8)
    return gradient

def main():
    # Initialize webcam capture
    cap = cv2.VideoCapture(0)
    
    # Check if webcam is opened successfully
    if not cap.isOpened():
        print("Cannot open webcam")
        return
    
    # Initialize Mediapipe Pose
    mp_pose = mp.solutions.pose
    pose = mp_pose.Pose()
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Flip frame horizontally for mirror effect
        frame = cv2.flip(frame, 1)
        
        # Convert the frame to RGB as Mediapipe uses RGB images
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Process the frame to detect pose landmarks
        result = pose.process(rgb_frame)
        
        # Create a mask for the person
        mask = np.zeros(frame.shape[:2], dtype=np.uint8)
        
        if result.pose_landmarks:
            # Get image dimensions
            img_h, img_w = frame.shape[:2]
            
            # Extract landmark points
            landmarks = result.pose_landmarks.landmark
            points = []
            for landmark in landmarks:
                x = int(landmark.x * img_w)
                y = int(landmark.y * img_h)
                points.append([x, y])
            
            # Convert points to a NumPy array
            points = np.array(points, dtype=np.int32)
            
            # Create a convex hull around the landmarks
            hull = cv2.convexHull(points)
            
            # Fill the convex hull to create the silhouette mask
            cv2.fillConvexPoly(mask, hull, 255)
            
            # Calculate the center of the person using moments
            M = cv2.moments(hull)
            if M['m00'] != 0:
                cX = int(M['m10'] / M['m00'])
                cY = int(M['m01'] / M['m00'])
            else:
                cX, cY = img_w // 2, img_h // 2  # Default to center if calculation fails
            
            # Create a radial gradient centered on the person
            gradient = create_radial_gradient(frame.shape[:2], center=(cX, cY))
            
            # Apply gradient to the mask
            gradient_mask = cv2.bitwise_and(gradient, gradient, mask=mask)
            
            # Convert gradient_mask to color using a colormap
            gradient_mask_color = cv2.applyColorMap(gradient_mask, cv2.COLORMAP_JET)
            
            # Add grunge effect by adding noise
            noise = np.random.randint(0, 50, gradient_mask_color.shape, dtype='uint8')
            gradient_mask_color = cv2.add(gradient_mask_color, noise)
            
            # Display the result
            cv2.imshow('Heat Signature Effect', gradient_mask_color)
        else:
            # If no person is detected, display a blank screen
            blank = np.zeros_like(frame)
            cv2.imshow('Heat Signature Effect', blank)
        
        # Press 'q' to exit
        if cv2.waitKey(30) & 0xFF == ord('q'):
            break
                
    # Release resources
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
