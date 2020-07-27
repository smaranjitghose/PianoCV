# Getting our dependencies
import cv2
import numpy as np
import time
import pygame
# Starting our session
pygame.init()
# Setting our screen size
w,h = 78,110
x = [10 + i*w for i in range(1,9)]
y = [10]*8

def piano(frame):
    '''
    Drawing our piano
    '''
    for i in range(0,8):
        cv2.rectangle(frame, (x[i], y[i]), (x[i] + w,y[i] + h), (255, 255, 255), -1)

    cv2.rectangle(frame,(x[0], y[0]),(x[7] + w, y[7] + h),(0,0,0),1)
    for i in range(1,8):
        cv2.line(frame, (x[i], y[i]), (x[i], y[i] + h), (0, 0, 0), 1)

def key_press(frame,x1,y1,w1,h1):
    '''
    Function to press the key of the piano and play our beat
    '''
    if x1 > x[0] and y1 > y[0] and x1 + w1 < x[0] + w and y1 + h1 < y[0] + h:
        cv2.rectangle(frame, (x[0], y[0]), (x[0] + w, y[0] + h), (211,211,211), -1)
        pygame.mixer.Sound('assets/a1.wav').play()
        time.sleep(0.10)
        pygame.mixer.Sound('assets/a1.wav').stop()

    elif x1>x[1] and y1 > y[1] and x1 + w1< x[1] + w and y1 + h1 < y[1] + h:
        cv2.rectangle(frame, (x[1], y[1]), (x[1] + w, y[1] + h), (211,211,211), -1)
        pygame.mixer.Sound('assets/b1.wav').play()
        time.sleep(0.10)
        pygame.mixer.Sound('assets/b1.wav').stop()
    elif x1 > x[2] and y1 > y[2] and x1 + w1< x[2] + w  and y1 + h1 < y[2] + h:
        cv2.rectangle(frame, (x[2], y[2]), (x[2] + w, y[2] + h), (211,211,211), -1)
        pygame.mixer.Sound('assets/c1.wav').play()
        time.sleep(0.10)
        pygame.mixer.Sound('assets/c1.wav').stop()
    elif x1 > x[3] and y1 > y[3] and x1 + w1 < x[3] + w and y1 + h1 < y[3]+h:
        cv2.rectangle(frame, (x[3], y[3]), (x[3] + w, y[3] + h), (211,211,211), -1)
        pygame.mixer.Sound('assets/c2.wav').play()
        time.sleep(0.10)
        pygame.mixer.Sound('assets/c2.wav').stop()
    elif x1 > x[4] and y1 > y[4] and x1 + w1 < x[4] + w and y1 + h1 < y[4] + h :
        cv2.rectangle(frame, (x[4], y[4]), (x[4] + w, y[4] + h), (211,211,211), -1)
        pygame.mixer.Sound('assets/d1.wav').play()
        time.sleep(0.10)
        pygame.mixer.Sound('assets/d1.wav').stop()
    elif x1 > x[5] and y1 > y[5] and x1 + w1< x[5] + w and y1 + h1 < y[5] + h:
        cv2.rectangle(frame, (x[5], y[5]), (x[5] + w, y[5] + h), (211,211,211), -1)
        pygame.mixer.Sound('assets/e1.wav').play()
        time.sleep(0.10)
        pygame.mixer.Sound('assets/e1.wav').stop()
    elif x1 > x[6] and y1 > y[6] and x1 + w1< x[6] + w and y + h1< y[6] + h:
        cv2.rectangle(frame, (x[6], y[6]), (x[6] + w, y[6] + h), (211,211,211), -1)
        pygame.mixer.Sound('assets/f1.wav').play()
        time.sleep(0.10)
        pygame.mixer.Sound('assets/f1.wav').stop()
    elif x1 > x[7] and y1 > y[7] and x1 + w1< x[7] + w and y1 + h1 < y[7] + h:
        cv2.rectangle(frame, (x[7], y[7]), (x[7] + w, y[7] + h), (211,211,211), -1)
        pygame.mixer.Sound('assets/g1.wav').play()
        time.sleep(0.10)
        pygame.mixer.Sound('assets/g1.wav').stop()

# Creating a VideoCapture object to read video from the primary camera
cap = cv2.VideoCapture(0)
while True:
    ret,frame = cap.read()
    frame = cv2.flip(frame, 1)
    frame =cv2.GaussianBlur(frame,(9,9),0)
    # Converting the color space from BGR to HSV as BGR is more sensitive to light
    frame_hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    # Drawing our piano
    piano(frame)
    # Generating mask for red range 1(0-10)
    lower_red = np.array([132, 90, 120])  
    upper_red = np.array([179, 255, 255])
    mask_1 = cv2.inRange(frame_hsv, lower_red, upper_red)
    # Generating mask for red range 2(170-180)
    lower_red = np.array([0, 110, 100])
    upper_red = np.array([3, 255, 255])
    mask_2 = cv2.inRange(frame_hsv, lower_red, upper_red)
    # Combining the masks obtained for both the ranges
    mask_f = mask_1 + mask_2
    # Creating kernels(sliders) of size 4x4 pixels and 15x15 pixels for boundary erosion and closing respectively
    kernel_1 = np.ones((4,4),np.uint8)
    kernel_2 = np.ones((15,15),np.uint8)
    # Perform Erosion on our mask
    mask_f = cv2.erode(mask_f,kernel_1,iterations = 1)
    # Removes false negatives
    mask_f = cv2.morphologyEx(mask_f,cv2.MORPH_CLOSE,kernel_2)
    xr, yr, wr, hr = 0, 0, 0, 0
    # Finding our contours
    contours, hierarchy = cv2.findContours(mask_f, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    try:
        for i in range(0,10):
            xr, yr, wr, hr = cv2.boundingRect(contours[i])
            if wr*hr > 1000:
                break
    except:
        pass
    cv2.rectangle(frame, (xr, yr), (xr + wr, yr + hr), (0, 0, 255), 2)
    key_press(frame, xr, yr, wr, hr)
    # Resizing our frame
    frame = cv2.resize(frame, (800, 800))
    # Display our main piano app
    cv2.imshow('Piano', frame)
    # Seeing how our masking actual works our
    # cv2.imshow('Mask',mask_f)
    # Check for keyboard input
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
