import cv2
import face_recognition
import mediapipe as mp
import pyautogui
import time

def count_fingers(lst):
    cnt = 0
    thresh = (lst.landmark[0].y - lst.landmark[9].y) / 2

    if (lst.landmark[5].y - lst.landmark[4].y) > 0.02:
        cnt += 1

    if (lst.landmark[9].y - lst.landmark[12].y) > thresh:
        cnt += 3

    return cnt

# Initialize Mediapipe and Face Recognition
mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands
hands = mp_hands.Hands()

# Screen dimensions for mouse control
screen_width, screen_height = pyautogui.size()

# Initialize webcam
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Error: Webcam not accessible.")
    exit()

# Load reference image for face recognition
reference_image_path = "D:/bindh/Pictures/Camera Roll/WIN_20241219_20_56_00_Pro.jpg"
try:
    reference_image = face_recognition.load_image_file(reference_image_path)
    reference_encoding = face_recognition.face_encodings(reference_image)[0]
except Exception as e:
    print(f"Error loading reference image: {e}")
    cap.release()
    exit()

while True:
    ret, frame = cap.read()
    if not ret:
        print("Error: Unable to read from webcam.")
        break

    # Face Recognition
    face_locations = face_recognition.face_locations(frame)
    face_encodings = face_recognition.face_encodings(frame, face_locations)

    for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
        matches = face_recognition.compare_faces([reference_encoding], face_encoding)

        if any(matches):
            cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
            print("Face matched!")

            # Hand Tracking
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = hands.process(rgb_frame)

            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

                    index_finger = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
                    x, y = int(index_finger.x * screen_width), int(index_finger.y * screen_height)
                    pyautogui.moveTo(x, y)

                    fingers_count = count_fingers(hand_landmarks)
                    if fingers_count == 1:
                        pyautogui.click()
                    elif fingers_count == 3:
                        pyautogui.rightClick()

    cv2.imshow("Face and Hand Tracking", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()