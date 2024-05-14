import cv2
import pygame
from pygame.locals import *
import pyaudio
import speech_recognition as sr
from moviepy.editor import VideoFileClip

# Initialize the webcam
cap = cv2.VideoCapture(0)

# Initialize pygame
pygame.init()
pygame.display.init()
pygame.mixer.init()
screen = pygame.display.set_mode((1920, 1080), FULLSCREEN, display=1)  # Use display index 2 for screen 2
clock = pygame.time.Clock()

# Load the video
video1 = VideoFileClip('ted.mp4')
pygame.mixer.music.load('HOW YA DOING - AUDIO FROM JAYUZUMI.COM.mp3')

# Load the pre-trained face detection classifier
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

show_video = False

# Initialize the speech recognizer
r = sr.Recognizer()

while True:
    # Capture frame-by-frame
    ret, frame = cap.read()

    # Flip the frame horizontally
    frame = cv2.flip(frame, 1)

    # Convert the frame to grayscale for face detection
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces in the grayscale frame
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    # Check if a person is detected
    if len(faces) > 0:
        if not show_video:
            show_video = True
            print("Person detected")
            # Play the MP3 file
            pygame.mixer.music.play()

            # Wait for the sound to finish playing
            while pygame.mixer.music.get_busy():
                pygame.time.Clock().tick(10)
            ted_playing = 0
            while not ted_playing:
                with sr.Microphone() as source:
                    print("Say something!")
                    audio = r.listen(source)
                try:
                    speech = r.recognize_google(audio)
                    print("You said: " + speech)
                    if "video" in speech.lower() or "ted" in speech.lower():
                        ted_playing = 1
                        video1.preview()
                except sr.UnknownValueError:
                    print("Could not understand audio")
                except sr.RequestError as e:
                    print("Could not request results; {0}".format(e))

    # Display the frame on the screen
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    frame = pygame.surfarray.make_surface(frame)
    screen.blit(frame, (0, 0))
    pygame.display.flip()
    clock.tick(30)

    # Check for quit events
    for event in pygame.event.get():
        if event.type == QUIT:
            pygame.quit()
            cap.release()
            cv2.destroyAllWindows()
            exit()

# Release the webcam
cap.release()
cv2.destroyAllWindows()
