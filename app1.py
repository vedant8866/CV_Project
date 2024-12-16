import cv2
import mediapipe as mp
import pyautogui
import time
import pywhatkit as kit
import streamlit as st
from threading import Thread
import threading

# Gesture functions
def count_fingers(lst):
    cnt = 0
    thresh = (lst.landmark[0].y * 100 - lst.landmark[9].y * 100) / 2
    if (lst.landmark[5].y * 100 - lst.landmark[8].y * 100) > thresh:  # Index finger
        cnt += 1
    if (lst.landmark[9].y * 100 - lst.landmark[12].y * 100) > thresh:  # Middle finger
        cnt += 1
    if (lst.landmark[13].y * 100 - lst.landmark[16].y * 100) > thresh:  # Ring finger
        cnt += 1
    if (lst.landmark[17].y * 100 - lst.landmark[20].y * 100) > thresh:  # Pinky finger
        cnt += 1
    if (lst.landmark[5].x * 100 - lst.landmark[4].x * 100) > 6:  # Thumb
        cnt += 1
    return cnt

def well_done(lst):
    thumb_up = (lst.landmark[4].y < lst.landmark[3].y < lst.landmark[2].y)
    pinky_up = (lst.landmark[20].y < lst.landmark[19].y)
    all_other_fingers_down = all(lst.landmark[tip].y > lst.landmark[base].y
                                 for tip, base in [(8, 6), (12, 10), (16, 14)])
    return thumb_up and pinky_up and all_other_fingers_down

def send_whatsapp_message():
    try:
        kit.sendwhatmsg_instantly("+918468813811", "Hii, how are you doing?", 10)
        time.sleep(10)  # Wait for WhatsApp Web to load
        pyautogui.press("enter")  # Simulate pressing Enter to send the message
    except Exception as e:
        print(f"Error sending message: {e}")

# Gesture detection function
def gesture_detection(running):
    cap = cv2.VideoCapture(0)
    drawing = mp.solutions.drawing_utils
    hands = mp.solutions.hands
    hand_obj = hands.Hands(max_num_hands=1)

    start_init = False
    prev = -1

    while running:
        _, frm = cap.read()
        frm = cv2.flip(frm, 1)
        res = hand_obj.process(cv2.cvtColor(frm, cv2.COLOR_BGR2RGB))

        if res.multi_hand_landmarks:
            hand_keyPoints = res.multi_hand_landmarks[0]
            if well_done(hand_keyPoints):
                send_whatsapp_message()
                time.sleep(3)  # Prevent repeated triggering

            cnt = count_fingers(hand_keyPoints)
            if cnt != prev:
                if cnt == 1:
                    pyautogui.press("right")
                elif cnt == 2:
                    pyautogui.press("left")
                elif cnt == 3:
                    pyautogui.press("up")
                elif cnt == 4:
                    pyautogui.press("down")
                elif cnt == 5:
                    pyautogui.press("space")
                prev = cnt
            drawing.draw_landmarks(frm, hand_keyPoints, hands.HAND_CONNECTIONS)

        cv2.imshow("Gesture Detection", frm)

        if cv2.waitKey(1) & 0xFF == 27:
            break

    cap.release()
    cv2.destroyAllWindows()

# Streamlit interface
st.title("Gesture Detection App")

running = False
detection_thread = None

def start_detection():
    global running, detection_thread
    running = True
    detection_thread = threading.Thread(target=gesture_detection, args=(running,))
    detection_thread.start()
    st.success("Gesture Detection Started!")

def stop_detection():
    global running, detection_thread
    if detection_thread is not None:
        running = False
        detection_thread.join()  # Wait for the thread to finish
        detection_thread = None  # Reset the thread
        st.success("Gesture Detection Stopped!")
    else:
        st.warning("No gesture detection is running.")

# Add buttons for control
if st.button("Start Detection"):
    start_detection()

if st.button("Stop Detection"):
    stop_detection()
