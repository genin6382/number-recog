import cv2
import mediapipe as mp

# Load the hand detection model from MediaPipe
def load_hand_detection_model():
    return mp.solutions.hands.Hands()

# Function to count the number of extended fingers
def count_extended_fingers(hand_landmarks):
    # Define the finger landmark indices
    finger_tip_indices = [4, 8, 12, 16, 20]  # Index of the fingertip landmarks

    # Initialize finger count
    finger_count = 0

    # Check each finger
    for finger_tip_index in finger_tip_indices:
        # Get the y-coordinate of the fingertip landmark
        finger_tip_y = hand_landmarks.landmark[finger_tip_index].y

        # Get the y-coordinate of the landmark closest to the fingertip (in this case, the middle finger base)
        middle_finger_base_y = hand_landmarks.landmark[finger_tip_index - 2].y

        # Check if the fingertip is above the middle finger base (extended)
        if finger_tip_y < middle_finger_base_y:
            finger_count += 1

    return finger_count

# Main function
def main():
    # Load the hand detection model
    hand_detection_model = load_hand_detection_model()

    # Initialize video capture
    cap = cv2.VideoCapture(0)  # Use 0 for default webcam

    while cap.isOpened():
        # Read frame from the webcam
        ret, frame = cap.read()
        if not ret:
            print("Failed to capture frame")
            break

        # Convert the frame to RGB
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Detect hands in the frame
        results = hand_detection_model.process(frame_rgb)

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                # Draw hand landmarks on the frame
                mp.solutions.drawing_utils.draw_landmarks(
                    frame, hand_landmarks, mp.solutions.hands.HAND_CONNECTIONS)

                # Count the number of extended fingers
                finger_count = count_extended_fingers(hand_landmarks)

                # Display the finger count on the frame
                cv2.putText(frame, f"Finger Count: {finger_count}", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        # Display the frame
        cv2.imshow('Finger Count', frame)

        # Check for key press to exit
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release the video capture and close all windows
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
