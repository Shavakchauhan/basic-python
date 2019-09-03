import cv2

face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')#we need to downloadv this files
eye_cascade = cv2.CascadeClassifier('haarcascade_eye.xml')
cap = cv2.VideoCapture(0) #here value is stored in  cap and taking value throug webcam  and 0 is coordinates

while True:
    ret, img = cap.read() #this function returns two values first is true for ret and second is pixel matrix for img
    print(ret)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) #it sets the color and the threshold
    faces = face_cascade.detectMultiScale(gray, 1.3, 5) #it sets the fps(frames per second) and for detecting more number of the people
    for (x, y, w, h) in faces:
        cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 5) # here (255,0,0) is the color coordinates and 5 is the width of rectangle
        roi_gray = gray[y:y + h, x:x + w]
        roi_color = img[y:y + h, x:x + w]

        eyes = eye_cascade.detectMultiScale(roi_gray)
        for (ex, ey, ew, eh) in eyes:
            cv2.rectangle(roi_color, (ex, ey), (ex + ew, ey + eh), (0, 255, 0), 2)

    cv2.imshow('img', img) #it shows us the output
    k = cv2.waitKey(30) & 0xff # iys waits for the user to press the key
    if k == 27: #ascii value of escape key
        break

cap.release() # it release the cap key
cv2.destroyAllWindows() # it destroy all
