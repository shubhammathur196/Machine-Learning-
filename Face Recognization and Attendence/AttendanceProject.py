# Importing
import numpy as np
import cv2
import os
import face_recognition
from datetime import datetime
# Path of Folder with images
path = 'ImagesAttendance'
# A list of the images to import from folder
images = []
# An array of the names of the images
classNames = []
# Specifying the path to myList
myList = os.listdir(path)
#print(myList)

for cl in myList:
    # Reading the Current Image
    curImg = cv2.imread(f'{path}/{cl}')
    # Appending the images
    images.append(curImg)
    # Appending the ClassNames
    classNames.append(os.path.splitext(cl)[0])
print(classNames)

# Encoding Process

# The function finds all the encodings of images
def findEncodings(images):
    # Creating an empty list
    encodeList = []
    # Converting to RGB
    for img in images:
        # Converting to RGB
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        # Finding the Encoding of the images
        encode = face_recognition.face_encodings(img)[0]
        # Appending the encoding to the list
        encodeList.append(encode)
    return encodeList



# Making a function for attendance
def markAttendance(name):
    with open('Attendance.csv','r+') as f:
        # Don;t want to repeat the names
        myDataList = f.readlines()
        nameList = []
        for line in myDataList:
            # Comma separated values
            entry = line.split(',')
            nameList.append(entry[0])
        if name not in nameList:
            now = datetime.now()
            dtString = now.strftime('%H:%M:%S')
            f.writelines(f'\n{name},{dtString}')


markAttendance('Elon')

# Calling the function
encodeListKnown = findEncodings(images)
print('Encoding Complete')

# Capturing Image from webcam
cap = cv2.VideoCapture(0)

while True:
    success, img = cap.read()
    imgS = cv2.resize(img, (0,0), None, 0.25, 0.25)
    imgS = cv2.cvtColor(imgS, cv2.COLOR_BGR2RGB)
    # Faces location in the current frame
    facesCurFrame = face_recognition.face_locations(imgS)
    # Encoding of face in the current frame
    encodeCurFrame = face_recognition.face_encodings(imgS, facesCurFrame)

    # Loop to encode current frame
    for encodeFace, faceLoc in zip(encodeCurFrame,facesCurFrame):
        # Comparing faces
        match = face_recognition.compare_faces(encodeListKnown, encodeFace)
        # Face Distance encoding
        faceDis = face_recognition.face_distance(encodeListKnown, encodeFace)

        #print(faceDis)
        matchIndex = np.argmin(faceDis)
        #
        if match[matchIndex]:
            name = classNames[matchIndex].upper()
            #print(name)
            # Face location
            y1, x2, y2, x1 = faceLoc
            # Multiplying the faceloc to 4 to make the rectangle accurate
            y1, x2, y2, x1 = y1*4, x2*4, y2*4, x1*4
            # Making a rectangle for the face to recognize
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.rectangle(img, (x1, y2-35), (x2, y2), (0, 255, 0), cv2.FILLED)
            cv2.putText(img, name, (x1+6, y2-6), cv2.FONT_HERSHEY_DUPLEX, 1, (255, 255, 255), 2)
            markAttendance(name)

    cv2.imshow('Webcam',img)
    cv2.waitKey(1)
    
    
    
    
    
# faceLoc= face_recognition.face_locations(imgElon)[0]
# encodeElon = face_recognition.face_encodings(imgElon)[0]
# # Making a Rectangle at Face location
# cv2.rectangle(imgElon,(faceLoc[3],faceLoc[0]),(faceLoc[1],faceLoc[2]),(255,0,255),2)
#
# faceLocTest= face_recognition.face_locations(imgTest)[0]
# encodeTest = face_recognition.face_encodings(imgTest)[0]
# # Making a Rectangle at Face location
# cv2.rectangle(imgTest,(faceLocTest[3],faceLocTest[0]),(faceLocTest[1],faceLocTest[2]),(255,0,255),2)
#
# Results= face_recognition.compare_faces([encodeElon],encodeTest)
# faceDist = face_recognition.face_distance([encodeElon],encodeTest)

