import face_recognition
import cv2
#import numpy as np
'''cap = cv2.VideoCapture(0)
while (True):
    ret, frame = cap.read()        
    key = cv2.waitKey(20)
    cv2.imshow("preview", frame)
    if key == 27: # exit on ESC
        break
cap.release()
cv2.destroyAllWindows()'''
known_image = face_recognition.load_image_file("14.jpg")
unknown_image = face_recognition.load_image_file("mm.jpg")
face_locations = face_recognition.face_locations(unknown_image) # detects all the faces in image
t = len(face_locations)
print(len(face_locations))
print(face_locations)
'''face_landmarks_list = face_recognition.face_landmarks("known_image")
print(face_landmarks_list)'''
for i in range(t) :
	biden_encoding = face_recognition.face_encodings(known_image)[0]
	unknown_encoding = face_recognition.face_encodings(unknown_image)[i]	
	results = face_recognition.compare_faces([biden_encoding], unknown_encoding)
	if (results == True):
		print("Present")
	



