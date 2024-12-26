import mediapipe as mp
import cv2
import pyautogui
import time

# Function to count fingers
def count_fingers(lst):
    cnt = 0
    thresh = (lst.landmark[0].y * 100 - lst.landmark[9].y * 100) / 2

    if (lst.landmark[5].y * 100 - lst.landmark[8].y * 100) > thresh:
        cnt += 1
    if (lst.landmark[9].y * 100 - lst.landmark[12].y * 100) > thresh:
        cnt += 2
    if (lst.landmark[13].y * 100 - lst.landmark[16].y * 100) > thresh:
        cnt += 3
    if (lst.landmark[17].y * 100 - lst.landmark[20].y * 100) > thresh:
        cnt += 4
    if (lst.landmark[5].x * 100 - lst.landmark[4].x * 100) > 6:
        cnt += 5

    return cnt

# Main Program
a = eval(input("Enter the number of hands (1 or 2): "))
if a in [1, 2]:
    mp_drawing = mp.solutions.drawing_utils
    mp_hands = mp.solutions.hands
    cap = cv2.VideoCapture(0)
    hand_obj = mp_hands.Hands(max_num_hands=a, min_detection_confidence=0.8, min_tracking_confidence=0.5)

    start_init = False
    prev = -1

    while True:
        end_time = time.time()
        _, frm = cap.read()
        frm = cv2.flip(frm, 1)

        res = hand_obj.process(cv2.cvtColor(frm, cv2.COLOR_BGR2RGB))

        if res.multi_hand_landmarks:
            hand_keyPoints = res.multi_hand_landmarks[0]
            cnt = count_fingers(hand_keyPoints)

            if prev != cnt:
                if not start_init:
                    start_time = time.time()
                    start_init = True
                elif (end_time - start_time) > 0.2:
                    actions = {
                        1: "right", 2: "left", 3: "up", 4: "down",
                        5: "space", 6: "enter", 7: "a", 8: "b",
                        9: ["l", "o", "v", "e"], 10: "d", 11: "e",
                        12: "f", 13: "g", 14: "h", 15: "i"
                    }

                    if cnt in actions:
                        action = actions[cnt]
                        if isinstance(action, list):
                            for key in action:
                                pyautogui.press(key)
                        else:
                            pyautogui.press(action)

                    prev = cnt
                    start_init = False

            mp_drawing.draw_landmarks(frm, hand_keyPoints, mp_hands.HAND_CONNECTIONS)

        cv2.imshow("Hand Tracking", frm)
        if cv2.waitKey(1) == 27:  # Exit on pressing 'Esc'
            cv2.destroyAllWindows()
            cap.release()
            break
else:
    print("ERROR: Number of hands exceeds limit (1 or 2 allowed)!")
