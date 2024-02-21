
import cv2
import time
import numpy as np
from tracker import *

tracker = EuclideanDistTracker()

def motion_detection():
    video_capture = cv2.VideoCapture(0)  # Initialize VideoCapture with camera index 0
    object_detector = cv2.createBackgroundSubtractorMOG2(history=100, varThreshold=40)
    time.sleep(2) #czas kamery na inicjację

    while True:
        ret, frame = video_capture.read()  # Read a frame klatka z kamery prezypisana do zmiennych
        
        mask = object_detector.apply(frame)
        _, mask = cv2.threshold(mask, 254, 255, cv2.THRESH_BINARY) #remove everything below 254
        countours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        
        # gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) #konwertowanie do szarości
        # blur = cv2.GaussianBlur(gray, (5, 5), 0) #rozmycie
        # ret, thresh = cv2.threshold(blur, 127, 255, cv2.THRESH_BINARY_INV) #obraz binarny

        # cnts = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE) #kontury na obrazie binarnym
        # cnts = imutils.grab_contours(cnts) #pobieranie konturu

        detection = []

        for c in countours:
            #area - remove small elements
            area = cv2.contourArea(c)
            if area > 100:
                # cv2.drawContours(frame, [c], -1, (0, 255, 0), 2)
                x, y, w, h = cv2.boundingRect(c)
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 3)

                detection.append([x, y, w, h])

        box_id = tracker.update(detection)

        for boxes_id in box_id:
            x, y, w, h, id = boxes_id
            cv2.putText(frame, str(id), (x, y-15), cv2.FONT_HERSHEY_PLAIN, 2, (255, 0, 0), 2)

        cv2.imshow("Motion Detection", frame)  # Display the frame
        cv2.imshow("Maska", mask)

        if cv2.waitKey(1) & 0xFF == ord('q'):  # Press 'q' to exit
            break

    video_capture.release()
    cv2.destroyAllWindows()

motion_detection()