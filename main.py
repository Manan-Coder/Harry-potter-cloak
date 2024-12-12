
import numpy as np
import cv2
import time


cap = cv2.VideoCapture(0)

time.sleep(3)

background = 0


for i in range(60):
    ret, background = cap.read()
background = np.flip(background, axis=1)

while cap.isOpened():
    ret, img = cap.read()
    if not ret:
        break
    img = np.flip(img, axis=1)

    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    lower_black = np.array([0, 0, 0])
    upper_black = np.array([180, 255, 50])


    mask1 = cv2.inRange(hsv, lower_black, upper_black)


    mask1 = cv2.morphologyEx(mask1, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8), iterations=2)
    mask1 = cv2.morphologyEx(mask1, cv2.MORPH_DILATE, np.ones((3, 3), np.uint8), iterations=1)


    mask2 = cv2.bitwise_not(mask1)


    res1 = cv2.bitwise_and(background, background, mask=mask1)
    res2 = cv2.bitwise_and(img, img, mask=mask2)
    final_output = cv2.addWeighted(res1, 1, res2, 1, 0)


    cv2.imshow('Invisible Cloak', final_output)


    k = cv2.waitKey(10)
    if k == 27:
        break


cap.release()
cv2.destroyAllWindows()
