import cv2
import mediapipe as mp
import pygame
from pygame.locals import DOUBLEBUF, OPENGL
import numpy as np
from OpenGL.GL import *
from OpenGL.GLU import *
import sys

# Initialize Pygame and set up display window with OpenGL
pygame.init()

# Define the screen size
SCREEN_WIDTH, SCREEN_HEIGHT = 640, 480
screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT), DOUBLEBUF | OPENGL)
pygame.display.set_caption('PS1 Style 3D Model Overlay')

# Initialize OpenGL
def init_opengl():
    glViewport(0, 0, SCREEN_WIDTH, SCREEN_HEIGHT)
    glMatrixMode(GL_PROJECTION)
    glLoadIdentity()
    gluPerspective(45, (SCREEN_WIDTH / SCREEN_HEIGHT), 0.1, 100.0)
    glMatrixMode(GL_MODELVIEW)
    glEnable(GL_DEPTH_TEST)
    glClearColor(0.0, 0.0, 0.0, 1.0)

init_opengl()

# Initialize MediaPipe Pose
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5,
    model_complexity=1
)

# Initialize webcam
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, SCREEN_WIDTH)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, SCREEN_HEIGHT)

# Define a simple humanoid model using vertices and faces
def create_humanoid_model():
    # Vertices for a simple humanoid model
    vertices = [
        # Head (cube)
        [-0.5, 1.5, 0.5],
        [0.5, 1.5, 0.5],
        [0.5, 1.5, -0.5],
        [-0.5, 1.5, -0.5],
        [-0.5, 2.5, 0.5],
        [0.5, 2.5, 0.5],
        [0.5, 2.5, -0.5],
        [-0.5, 2.5, -0.5],

        # Body (cube)
        [-0.75, 0.0, 0.5],
        [0.75, 0.0, 0.5],
        [0.75, 0.0, -0.5],
        [-0.75, 0.0, -0.5],
        [-0.75, 1.5, 0.5],
        [0.75, 1.5, 0.5],
        [0.75, 1.5, -0.5],
        [-0.75, 1.5, -0.5],

        # Left Arm (cube)
        [-1.25, 0.0, 0.25],
        [-0.75, 0.0, 0.25],
        [-0.75, 0.0, -0.25],
        [-1.25, 0.0, -0.25],
        [-1.25, 1.25, 0.25],
        [-0.75, 1.25, 0.25],
        [-0.75, 1.25, -0.25],
        [-1.25, 1.25, -0.25],

        # Right Arm (cube)
        [0.75, 0.0, 0.25],
        [1.25, 0.0, 0.25],
        [1.25, 0.0, -0.25],
        [0.75, 0.0, -0.25],
        [0.75, 1.25, 0.25],
        [1.25, 1.25, 0.25],
        [1.25, 1.25, -0.25],
        [0.75, 1.25, -0.25],

        # Left Leg (cube)
        [-0.5, -1.5, 0.25],
        [0.0, -1.5, 0.25],
        [0.0, -1.5, -0.25],
        [-0.5, -1.5, -0.25],
        [-0.5, 0.0, 0.25],
        [0.0, 0.0, 0.25],
        [0.0, 0.0, -0.25],
        [-0.5, 0.0, -0.25],

        # Right Leg (cube)
        [0.0, -1.5, 0.25],
        [0.5, -1.5, 0.25],
        [0.5, -1.5, -0.25],
        [0.0, -1.5, -0.25],
        [0.0, 0.0, 0.25],
        [0.5, 0.0, 0.25],
        [0.5, 0.0, -0.25],
        [0.0, 0.0, -0.25],
    ]

    # Faces for cubes (each cube has 12 triangles)
    faces = []

    # Function to add cube faces given the starting index of vertices
    def add_cube_faces(start_idx):
        # Each face is two triangles
        cube_faces = [
            [start_idx, start_idx+1, start_idx+2],
            [start_idx, start_idx+2, start_idx+3],  # Bottom face
            [start_idx+4, start_idx+5, start_idx+6],
            [start_idx+4, start_idx+6, start_idx+7],  # Top face
            [start_idx, start_idx+1, start_idx+5],
            [start_idx, start_idx+5, start_idx+4],  # Front face
            [start_idx+1, start_idx+2, start_idx+6],
            [start_idx+1, start_idx+6, start_idx+5],  # Right face
            [start_idx+2, start_idx+3, start_idx+7],
            [start_idx+2, start_idx+7, start_idx+6],  # Back face
            [start_idx+3, start_idx+0, start_idx+4],
            [start_idx+3, start_idx+4, start_idx+7],  # Left face
        ]
        faces.extend(cube_faces)

    # Add faces for head
    add_cube_faces(0)
    # Add faces for body
    add_cube_faces(8)
    # Add faces for left arm
    add_cube_faces(16)
    # Add faces for right arm
    add_cube_faces(24)
    # Add faces for left leg
    add_cube_faces(32)
    # Add faces for right leg
    add_cube_faces(40)

    return vertices, faces

# Create the humanoid model
vertices, faces = create_humanoid_model()

# Function to render the 3D model
def render_model(vertices, faces):
    glBegin(GL_TRIANGLES)
    for face in faces:
        for vertex_i in face:
            glColor3f(1.0, 1.0, 1.0)  # White color for the model
            glVertex3fv(vertices[vertex_i])
    glEnd()

# Function to get pose landmarks
def get_pose_landmarks(results):
    if not results.pose_landmarks:
        return None

    landmarks = results.pose_landmarks.landmark
    pose_landmarks = {}
    for idx, landmark in enumerate(landmarks):
        pose_landmarks[idx] = (landmark.x, landmark.y, landmark.z)
    return pose_landmarks

# Function to update the model transformation based on pose landmarks
def update_model_transformation(pose_landmarks):
    glLoadIdentity()
    glTranslatef(0.0, 0.0, -6.0)  # Move the model away from the camera

    if pose_landmarks is None:
        return

    # Use shoulder points to rotate the model
    left_shoulder = pose_landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value]
    right_shoulder = pose_landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value]

    # Calculate the angle between shoulders
    delta_x = right_shoulder[0] - left_shoulder[0]
    delta_y = right_shoulder[1] - left_shoulder[1]
    angle = np.degrees(np.arctan2(delta_y, delta_x))

    # Use nose position to position the model
    nose = pose_landmarks[mp_pose.PoseLandmark.NOSE.value]
    x = (nose[0] - 0.5) * 4  # Map from [0,1] to [-2,2]
    y = (0.5 - nose[1]) * 4  # Invert y-axis and map to [-2,2]

    glTranslatef(x, y, 0.0)
    glRotatef(-angle, 0.0, 1.0, 0.0)  # Rotate around y-axis
    glScalef(0.5, 0.5, 0.5)  # Scale the model

# Function to draw the camera feed as background
def draw_camera_feed(frame):
    # Convert the frame to a texture
    frame = cv2.flip(frame, 0)  # OpenGL expects the texture to be flipped vertically
    frame_data = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB).tobytes()

    glMatrixMode(GL_PROJECTION)
    glPushMatrix()
    glLoadIdentity()
    glOrtho(0, SCREEN_WIDTH, 0, SCREEN_HEIGHT, -1, 1)
    glMatrixMode(GL_MODELVIEW)
    glLoadIdentity()
    glDisable(GL_DEPTH_TEST)

    glEnable(GL_TEXTURE_2D)
    texture_id = glGenTextures(1)
    glBindTexture(GL_TEXTURE_2D, texture_id)

    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR)
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)

    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, SCREEN_WIDTH, SCREEN_HEIGHT, 0, GL_RGB, GL_UNSIGNED_BYTE, frame_data)

    # Draw a quad filling the screen
    glBegin(GL_QUADS)
    glTexCoord2f(0, 0)
    glVertex2f(0, 0)
    glTexCoord2f(1, 0)
    glVertex2f(SCREEN_WIDTH, 0)
    glTexCoord2f(1, 1)
    glVertex2f(SCREEN_WIDTH, SCREEN_HEIGHT)
    glTexCoord2f(0, 1)
    glVertex2f(0, SCREEN_HEIGHT)
    glEnd()

    glDeleteTextures([texture_id])
    glDisable(GL_TEXTURE_2D)
    glEnable(GL_DEPTH_TEST)
    glMatrixMode(GL_PROJECTION)
    glPopMatrix()
    glMatrixMode(GL_MODELVIEW)

# Main loop
def main():
    """Main loop for PS1-style 3D model overlay."""
    running = True
    clock = pygame.time.Clock()

    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        ret, frame = cap.read()
        if not ret:
            break

        # Flip and convert the frame
        frame = cv2.flip(frame, 1)
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Process the frame with MediaPipe
        results = pose.process(frame_rgb)
        pose_landmarks = get_pose_landmarks(results)

        # Clear buffers
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)

        # Draw the camera feed
        draw_camera_feed(frame)

        # Update model transformation based on pose
        update_model_transformation(pose_landmarks)

        # Render the 3D model
        render_model(vertices, faces)

        pygame.display.flip()
        clock.tick(30)

    cap.release()
    pygame.quit()
    pose.close()
    sys.exit()

if __name__ == '__main__':
    main()
