import pickle
import cv2
import mediapipe as mp
import numpy as np
from tkinter import Tk, Canvas, Entry, Button, PhotoImage, END
import threading
import time

# Load the pre-trained model
model_dict = pickle.load(open('./model.p', 'rb'))
model = model_dict['model']

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

hands = mp_hands.Hands(static_image_mode=True, min_detection_confidence=0.3)

labels_dict = {
    0: 'Where?', 1: 'Wait', 2: 'Stop', 3: 'Pay', 4: 'Left', 5: 'Right', 6: 'Pass(ticket)', 7: 'request', 8: 'please'
}

# Create a function to recognize characters from the camera and update the UI
def recognize_and_update_text():
    data_aux = []
    x_ = []
    y_ = []

    ret, frame = cap.read()

    H, W, _ = frame.shape

    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    results = hands.process(frame_rgb)
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(
                frame,  # image to draw
                hand_landmarks,  # model output
                mp_hands.HAND_CONNECTIONS,  # hand connections
                mp_drawing_styles.get_default_hand_landmarks_style(),
                mp_drawing_styles.get_default_hand_connections_style())

        for hand_landmarks in results.multi_hand_landmarks:
            for i in range(len(hand_landmarks.landmark)):
                x = hand_landmarks.landmark[i].x
                y = hand_landmarks.landmark[i].y

                x_.append(x)
                y_.append(y)

            for i in range(len(hand_landmarks.landmark)):
                x = hand_landmarks.landmark[i].x
                y = hand_landmarks.landmark[i].y
                data_aux.append(x - min(x_))
                data_aux.append(y - min(y_))

        x1 = int(min(x_) * W) - 10
        y1 = int(min(y_) * H) - 10

        x2 = int(max(x_) * W) - 10
        y2 = int(max(y_) * H) - 10

        prediction = model.predict([np.asarray(data_aux)])

        predicted_character = labels_dict[int(prediction[0])]

        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 0), 4)
        cv2.putText(frame, predicted_character, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 3.0, (0, 128, 128), 3,
                    cv2.LINE_AA)

        # Append the recognized character to the Entry widget in the UI
        entry_1.insert(END, predicted_character)

        # Delay for 5 seconds before recognizing the next character
        time.sleep(2)

    # Display the frame in a window
    cv2.imshow('frame', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        return

# Create a button to start recognizing characters
def generate_button_callback():
    # Schedule the initial recognition
    window.after(0, recognize_and_update_text)

# Initialize the camera
cap = cv2.VideoCapture(0)

# Create a separate thread to run the camera feed
def camera_feed_thread():
    while True:
        recognize_and_update_text()

# Start the camera feed thread
camera_thread = threading.Thread(target=camera_feed_thread)
camera_thread.daemon = True
camera_thread.start()

# Create the Tkinter UI
window = Tk()
window.geometry("862x519")
window.configure(bg="#3A7FF6")

canvas = Canvas(
    window,
    bg="#3A7FF6",
    height=519,
    width=862,
    bd=0,
    highlightthickness=0,
    relief="ridge"
)
canvas.place(x=0, y=0)
canvas.create_rectangle(
    430.9999999999999,
    0.0,
    861.9999999999999,
    519.0,
    fill="#FCFCFC",
    outline=""
)

# Create an Entry widget
# Create an Entry widget with a larger font and dimensions
entry_1 = Entry(
    bd=0,
    bg="#F1F5FF",
    fg="#000716",
    highlightthickness=0,
    font=("Roboto", 24),  # Increase the font size
)
entry_1.place(
    x=489.9999999999999,
    y=137.0,
    width=500,  # Increase the width
    height=80,  # Increase the height
)

# Create a button to start recognizing characters
button_image_1: PhotoImage = PhotoImage(file="C:\\Users\Saher\PycharmProjects\Kiosk_Sign\assets\frame0\button_1.png")
button_1 = Button(
    image=button_image_1,
    borderwidth=0,
    highlightthickness=0,
    command=generate_button_callback,  # Changed the command to generate_button_callback
    relief="flat"
)
button_1.place(
    x=575.9999999999999,
    y=292.0,
    width=180.0,
    height=55.0
)

canvas.create_text(
    36.999999999999886,
    66.0,
    anchor="nw",
    text="Sign Wave",
    fill="#FFFFFF",
    font=("Roboto Bold", 64 * -1)
)

window.resizable(False, False)
window.mainloop()

# Release the camera and close OpenCV windows when finished
cap.release()
cv2.destroyAllWindows()
