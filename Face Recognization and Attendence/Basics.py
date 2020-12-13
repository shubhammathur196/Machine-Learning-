# Importing Libraries
import cv2
import numpy as np
import face_recognition

# Loading image using Face recognition library
imgElon= face_recognition.load_image_file('C:\\Users\sm291\Desktop\Machine learning\\face\ImagesBasic\\Elon Musk.jpg')
# Changing the Image to RGB
imgElon= cv2.cvtColor(imgElon,cv2.COLOR_BGR2RGB)
imgTest = face_recognition.load_image_file('C:\\Users\sm291\Desktop\Machine learning\\face\ImagesBasic\\Elon Test.jpg')
imgTest= cv2.cvtColor(imgTest, cv2.COLOR_BGR2RGB)



# Recognizing the Location of the face in the image
faceLoc= face_recognition.face_locations(imgElon)[0]
# Encoding of the Face
encodeElon = face_recognition.face_encodings(imgElon)[0]
# Making a Rectangle at Face location passing the x1,y1,x1,y2
cv2.rectangle(imgElon,(faceLoc[3],faceLoc[0]),(faceLoc[1],faceLoc[2]),(255,0,255),2)

# Doing the same for the Test Image
faceLocTest= face_recognition.face_locations(imgTest)[0]
encodeTest = face_recognition.face_encodings(imgTest)[0]
# Making a Rectangle at Face location
cv2.rectangle(imgTest,(faceLocTest[3],faceLocTest[0]),(faceLocTest[1],faceLocTest[2]),(255,0,255),2)


# Comparing the Two faces of the two images
Results= face_recognition.compare_faces([encodeElon],encodeTest)
# Face distance or the 128 encoding
faceDist = face_recognition.face_distance([encodeElon],encodeTest)
# Printing the Results and Face Distance(more the distance more different the face is )
print(Results,faceDist)

# Putting a Text on the Test Image
cv2.putText(imgTest,f'{Results}{round(faceDist[0],2)}',(50,50),cv2.FONT_HERSHEY_SCRIPT_COMPLEX,1,(0,0,255),2)


# Display the Images
cv2.imshow('Elon Musk',imgElon)
cv2.imshow('Elon Test',imgTest)
cv2.waitKey(0)