from flask import Flask,render_template,Response,request
import cv2
import face_recognition_models
from flask.wrappers import Request
import face_recognition
import numpy as np
import os
import pygame
import datetime as dt

app=Flask(__name__,template_folder='templetes')

# Get a reference to webcam #0 (the default one)
video_capture = cv2.VideoCapture(0)

# Load a sample picture and learn how to recognize it.
obama_image = face_recognition.load_image_file("obama.jpg")
obama_face_encoding = face_recognition.face_encodings(obama_image)[0]

# Load a second sample picture and learn how to recognize it.
himanshu_image = face_recognition.load_image_file("himanshu_ratnaparkhi.jpg")
himanshu_face_encoding = face_recognition.face_encodings(himanshu_image)[0]

# Create arrays of known face encodings and their names
known_face_encodings = [
    obama_face_encoding,
    himanshu_face_encoding
]
known_face_names = [
    "Barack Obama",
    "Himanshu"
]

# Initialize some variables
face_locations = []
face_encodings = []
face_names = []
process_this_frame = True
kasim = "Kasim Baig"
unkown = "Unknown"
h1 = "Himanshu"
lasttime = dt.datetime.now()
currenttime = dt.datetime.now()
pygame.mixer.init()
pygame.mixer.set_num_channels(8)
voice = pygame.mixer.Channel(2)
sound = pygame.mixer.Sound("unknown.wav")
sound2 = pygame.mixer.Sound("dog-barking.wav")


def generate_frames():
    while True:
            
        ## read the camera frame
        success,frame=video_capture.read()

        if not success:
            break
        else:
            # Resize frame of video to 1/4 size for faster face recognition processing
            small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)

            # Convert the image from BGR color (which OpenCV uses) to RGB color (which face_recognition uses)
            rgb_small_frame = small_frame[:, :, ::-1]

            # Only process every other frame of video to save time
            # if process_this_frame:
                # Find all the faces and face encodings in the current frame of video
            face_locations = face_recognition.face_locations(rgb_small_frame)
            face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)

            face_names = []
            for face_encoding in face_encodings:
                    # See if the face is a match for the known face(s)
                    matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
                    name = "Unknown"

                    # # If a match was found in known_face_encodings, just use the first one.
                    # if True in matches:
                    #     first_match_index = matches.index(True)
                    #     name = known_face_names[first_match_index]

                    # Or instead, use the known face with the smallest distance to the new face
                    face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
                    best_match_index = np.argmin(face_distances)
                    if matches[best_match_index]:
                        name = known_face_names[best_match_index]

                    face_names.append(name)
                    print(face_names)
                    for i in face_names:
                        if i in unkown :
                            if voice.get_busy() == False:
                                #if (currenttime - lasttime).seconds > 3:
                                lasttime = dt.datetime.now()
                                voice.play(sound)
            
                    for i in face_names:
                        if i in h1 :
                            if voice.get_busy() == False:
                                #if (currenttime - lasttime).seconds > 3:
                                
                                lasttime = dt.datetime.now()
                                voice.play(sound2)


            # process_this_frame = not process_this_frame


            # Display the results
            for (top, right, bottom, left), name in zip(face_locations, face_names):
                # Scale back up face locations since the frame we detected in was scaled to 1/4 size
                top *= 4
                right *= 4
                bottom *= 4
                left *= 4

                # Draw a box around the face
                cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)

                # Draw a label with a name below the face
                cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (0, 0, 255), cv2.FILLED)
                font = cv2.FONT_HERSHEY_DUPLEX
                cv2.putText(frame, name, (left + 6, bottom - 6), font, 1.0, (255, 255, 255), 1)


            ret,buffer=cv2.imencode('.jpg',frame)
            frame=buffer.tobytes()
            yield(b'--frame\r\n'
                    b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')



@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video')
def video():
    print(generate_frames())

    return Response(generate_frames(),mimetype='multipart/x-mixed-replace; boundary=frame')


if __name__=="__main__":
    app.run(debug=True)