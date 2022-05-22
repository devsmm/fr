import cv2 as cv
import numpy as np
import face_recognition
import os
from datetime import datetime
import pyttsx3

""" 
Getting the images ready to be verified as it is stored in the Images_attendance repository.
"""
path = 'Images_attendance'
images = []
classNames = []
myList = os.listdir(path)

for current_list in myList:
    current_img = cv.imread(f'{path}/{current_list}')
    images.append(current_img)
    classNames.append(os.path.splitext(current_list)[0])

"""
Encoding every images in the repository in order to be compared.
"""


def findEncodings(images):
    encodeList = []
    for img in images:
        img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
        encodeImg = face_recognition.face_encodings(img)[0]
        encodeList.append(encodeImg)
    return encodeList


def markAttendance(name):
    with open('Attendance.csv', "r+") as f:
        myDataList = f.readlines()
        nameList = []
        for line in myDataList:
            entry = line.split(",")
            nameList.append(entry[0])
        if name not in nameList:
            now = datetime.now()
            dtString = now.strftime('%H:%M:%S')
            f.writelines(f'\n{name},{dtString}')


encodeListKnown = findEncodings(images)
print("Encoding list complete")

"""  
Getting the web cam ready to detect the faces.
"""
camWidth, camHeight = 640, 480
videoCapture = cv.VideoCapture(0)
videoCapture.set(3, camWidth)
videoCapture.set(4, camHeight)
engine = pyttsx3.init()
while True:
    success, img = videoCapture.read()
    img_sized = cv.resize(img, (0, 0), None, 0.25, 0.25)
    img_sized = cv.cvtColor(img_sized, cv.COLOR_BGR2RGB)

    """
    storing the location of the face from the image and encoding it.
    """
    facesCurFrame = face_recognition.face_locations(img_sized)
    encodeCurFrame = face_recognition.face_encodings(img_sized, facesCurFrame)

    """ 
    Comparing the faces and calculating the distance in the faces.
    """
    for encodeFace, faceLocation in zip(encodeCurFrame, facesCurFrame):
        matches = face_recognition.compare_faces(encodeListKnown, encodeFace)
        face_distance = face_recognition.face_distance(encodeListKnown, encodeFace)
        print(face_distance)
        matchIndex = np.argmin(face_distance)

        """
        If the faces match then the name is printed on the screen as well as the console.
        """
        if matches[matchIndex]:
            name = classNames[matchIndex].upper()
            engine.say("you are marked present")
            for n in name:
                engine.say(n)
            engine.runAndWait()
            print(name)
            y1, x2, y2, x1 = faceLocation
            y1, x2, y2, x1 = y1 * 4, x2 * 4, y2 * 4, x1 * 4
            cv.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv.rectangle(img, (x1, y2 - 35), (x2, y2), (255, 0, 255), cv.FILLED)
            cv.putText(img, name, (x1 + 6, y2 - 6), cv.FONT_HERSHEY_COMPLEX, 1, (255, 255, 255), 2)
            markAttendance(name)

    """ Waiting in the video camera for some face to be detected"""
    cv.imshow("attendance", img)
    cv.waitKey(1)
