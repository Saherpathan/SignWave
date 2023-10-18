import pickle
import cv2
import mediapipe as mp
import numpy as np
from tkinter import Tk, Canvas, Entry, Button, PhotoImage, Label, END
import threading
from PIL import Image, ImageTk

# Load the pre-trained model
model_dict = pickle.load(open('./model.p', 'rb'))
model = model_dict['model']

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

hands = mp_hands.Hands(static_image_mode=True, min_detection_confidence=0.3)

labels_dict = {
    0: 'Where?', 1: 'Wait', 2: 'Stop', 3: 'Pay', 4: 'Left', 5: 'Right', 6: 'Pass(ticket)', 7: 'Request', 8: 'Please'
}

# Initialize the recognized text variable
recognized_text = ""

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

        # Update the recognized text variable
        global recognized_text
        recognized_text = predicted_character

        # Append the recognized character to the Entry widget in the UI
        entry_1.delete(0, END)
        entry_1.insert(0, recognized_text)

    # Display the frame in a window
    cv2.imshow('frame', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        return

# Create a button to start recognizing characters
def generate_button_callback():
    # Clear the text in the Entry widget
    entry_1.delete(0, END)

    # Call recognize_and_update_text immediately and then schedule it to be called every 15 seconds
    recognize_and_update_text()

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
window.configure(bg="#1E1E1E")  # Dark grayish-black background

# Create a canvas with the same background color
canvas = Canvas(
    window,
    bg="#1E1E1E",
    height=519,
    width=862,
    bd=0,
    highlightthickness=0,
    relief="ridge"
)
canvas.place(x=0, y=0)

# Load and display a logo or image
logo_image = Image.open("sw.png")
logo_photo = ImageTk.PhotoImage(logo_image)
logo_label = Label(window, image=logo_photo, bg="#1E1E1E")
logo_label.photo = logo_photo
logo_label.place(x=10, y=10)

# Create a label with a description
description_label = Label(
    window,
    text="SignWave",  # Update the app name
    font=("Nourd", 36),
    fg="white",
    bg="#1E1E1E",
)
description_label.place(x=400, y=20)

# Create an Entry widget with a modern font and styling
entry_1 = Entry(
    bd=0,
    bg="#292929",  # Dark grayish background
    fg="white",
    highlightbackground="#666666",
    highlightcolor="#666666",
    font=("Nourd", 24),
)
entry_1.place(
    x=200,
    y=400,
    width=462,
    height=60,
)

# Create a button with a modern appearance
button_image_1: PhotoImage = PhotoImage(file="nn.png")
button_1 = Button(
    image=button_image_1,
    borderwidth=0,
    highlightthickness=5,
    command=generate_button_callback,
    relief="sunken",
)
button_1.place(
    x=680,
    y=400,
    width=162,
    height=60,
)

# Start the initial recognition
generate_button_callback()

window.resizable(False, False)
window.mainloop()
