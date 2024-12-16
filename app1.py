import cv2
import mediapipe as mp
import streamlit as st
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

# Gesture detection function
def gesture_detection(running, callback):
    cap = cv2.VideoCapture(0)
    drawing = mp.solutions.drawing_utils
    hands = mp.solutions.hands
    hand_obj = hands.Hands(max_num_hands=1)

    prev = -1

    while running():
        _, frm = cap.read()
        frm = cv2.flip(frm, 1)
        res = hand_obj.process(cv2.cvtColor(frm, cv2.COLOR_BGR2RGB))

        if res.multi_hand_landmarks:
            hand_keyPoints = res.multi_hand_landmarks[0]
            cnt = count_fingers(hand_keyPoints)

            if cnt != prev:
                callback(f"Detected {cnt} fingers.")
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
log = []

def start_detection():
    global running, detection_thread

    if not running:
        running = True
        detection_thread = threading.Thread(target=gesture_detection, args=(lambda: running, update_log))
        detection_thread.start()
        st.success("Gesture Detection Started!")
    else:
        st.warning("Gesture Detection is already running.")

def stop_detection():
    global running, detection_thread

    if running:
        running = False
        detection_thread.join()
        detection_thread = None
        st.success("Gesture Detection Stopped!")
    else:
        st.warning("No Gesture Detection is currently running.")

def update_log(message):
    log.append(message)

# Add buttons for control
if st.button("Start Detection"):
    start_detection()

if st.button("Stop Detection"):
    stop_detection()

# Display log
st.write("Log:")
for message in log[-5:]:  # Display the last 5 messages
    st.write(message)
