import numpy as np
import cv2

face_classifier = cv2.CascadeClassifier(r'C:\Users\Ariya Rayaneh\Desktop\haarcascade_frontalface_default.xml')
eye_classifier = cv2.CascadeClassifier(r'C:\Users\Ariya Rayaneh\Desktop\haarcascade_eye.xml')
mouth_classifier = cv2.CascadeClassifier(r'C:\Users\Ariya Rayaneh\Desktop\mouth.xml')

img = cv2.imread(r'C:\Users\Ariya Rayaneh\Desktop\download.jpg')
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
faces = face_classifier.detectMultiScale(gray, 1.3, 5)

for (x, y, w, h) in faces:
    cv2.rectangle(img, (x, y), (x + w, y + h), (127, 0, 255), 2)
    roi_gray = gray[y:y + h, x:x + w]
    roi_color = img[y:y + h, x:x + w]
    eyes = eye_classifier.detectMultiScale(roi_gray)
    mouth=mouth_classifier.detectMultiScale(roi_gray)

    for (ex, ey, ew, eh) in eyes:
        cv2.rectangle(roi_color, (ex, ey), (ex + ew, ey + eh), (255, 255, 0), 2)

    for (ex1, ey1, ew1, eh1) in mouth:
        cv2.rectangle(roi_color, (ex1, ey1), (ex1 + ew1, ey1 + eh1), (255, 255, 0), 2)

cv2.imwrite(r'C:\Users\Ariya Rayaneh\Desktop\download101.jpg', img)
cv2.waitKey()

cv2.destroyAllWindows()




