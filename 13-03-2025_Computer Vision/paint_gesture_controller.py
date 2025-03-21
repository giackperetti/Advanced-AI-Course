import cv2
import math
import mediapipe as mp
import pyautogui
from datetime import datetime as dt
import time

# Disable pyautogui fail-safe
pyautogui.FAILSAFE = False

def resize(image):
    DESIRED_HEIGHT = 480
    DESIRED_WIDTH = 480
    h, w = image.shape[:2]
    if h < w:
        img = cv2.resize(image, (DESIRED_WIDTH, math.floor(h / (w / DESIRED_WIDTH))))
    else:
        img = cv2.resize(image, (math.floor(w / (h / DESIRED_HEIGHT)), DESIRED_HEIGHT))
    return img

def press_mouse_down():
    x, y = pyautogui.position()
    pyautogui.mouseDown(x=x, y=y)

def release_mouse():
    pyautogui.mouseUp()

# Initialize MediaPipe
BaseOptions = mp.tasks.BaseOptions
GestureRecognizer = mp.tasks.vision.GestureRecognizer
GestureRecognizerOptions = mp.tasks.vision.GestureRecognizerOptions
VisionRunningMode = mp.tasks.vision.RunningMode

base_options = BaseOptions(
    model_asset_path="./gesture_recognizer.task", delegate=BaseOptions.Delegate.CPU
)
options = GestureRecognizerOptions(
    base_options=base_options, running_mode=VisionRunningMode.VIDEO, num_hands=1
)
recognizer = GestureRecognizer.create_from_options(options)

cap = cv2.VideoCapture(0)
ret, frame = cap.read()
cv2.imshow("Gesture Controller", cv2.flip(frame, 1))
cv2.waitKey(1)

try:
    pyautogui.hotkey('command', 'space')
    time.sleep(0.5)
    pyautogui.write('arc')
    time.sleep(0.5)
    pyautogui.press('enter')
    time.sleep(2)
    pyautogui.hotkey('fn', 'control', 'f')
    time.sleep(0.5)
except Exception as e:
    print(f"Error opening Arc: {e}")

try:
    # pyautogui.hotkey('alt', 'tab')
    time.sleep(0.5)
except Exception as e:
    print(f"Error focusing camera window: {e}")

screen_width, screen_height = pyautogui.size()

is_drawing = False
previous_x, previous_y = 0, 0
draw_counter = 0

timestamp = 0
while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to capture image")
        break

    frame = cv2.flip(frame, 1)
    image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame)

    timestamp += 33
    recognition_result = recognizer.recognize_for_video(image, timestamp)
    h, w = frame.shape[:2]

    if is_drawing:
        cv2.circle(frame, (30, 30), 15, (0, 255, 0), -1)
    else:
        cv2.circle(frame, (30, 30), 15, (0, 0, 255), -1)

    if recognition_result.gestures and recognition_result.hand_landmarks:
        gesture = recognition_result.gestures[0][0]
        hand = recognition_result.handedness[0][0]
        landmarks = recognition_result.hand_landmarks[0]

        index_finger_tip = landmarks[8]
        x_pos = int(index_finger_tip.x * w)
        y_pos = int(index_finger_tip.y * h)

        screen_x = int(index_finger_tip.x * screen_width)
        screen_y = int(index_finger_tip.y * screen_height)

        cv2.circle(frame, (x_pos, y_pos), 10, (255, 0, 0), -1)

        print(f"{dt.now()}: {gesture.category_name}: {round(gesture.score * 100, 2)} - {hand.display_name}: {round(hand.score * 100, 2)}")

        if gesture.category_name == "Pointing_Up" and not is_drawing:
            is_drawing = True
            pyautogui.moveTo(screen_x, screen_y)
            press_mouse_down()
            previous_x, previous_y = screen_x, screen_y

        elif gesture.category_name == "Closed_Fist" and is_drawing:
            is_drawing = False
            release_mouse()

        elif gesture.category_name == "ILoveYou":
            print("ILoveYou gesture detected - Exiting program")
            if is_drawing:
                release_mouse()
            break

        if is_drawing:
            smooth_x = (screen_x + previous_x) // 2
            smooth_y = (screen_y + previous_y) // 2

            pyautogui.moveTo(smooth_x, smooth_y)
            press_mouse_down()

            draw_counter += 1
            previous_x, previous_y = smooth_x, smooth_y

    cv2.imshow("Gesture Controller", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        if is_drawing:
            release_mouse()
        break

recognizer.close()
cap.release()
cv2.destroyAllWindows()
