import cv2
from display import Display
import numpy as np
from extractor import Extractor
import warnings
warnings.filterwarnings('ignore')

w = 1164//2
h = 874//2
F = 1

disp = Display(w,h)

k = np.array([[F,0,w//2],[0,F,h//2],[0,0,1]])


fe = Extractor(k)


def process_frame(img):

    img = cv2.resize(img, (w,h))
    matches = fe.extract(img)
    print("%d matches" %(len(matches)))


    for pt1,pt2 in matches:
        u1,v1 = fe.denormalize(pt1)
        u2,v2 = fe.denormalize(pt2)


        cv2.circle(img, (u1,v1), color = (0,255,0), radius = 3)
        cv2.line(img, (u1,v1),(u2,v2), color = (255,0,0))




    disp.paint(img)




if __name__ == "__main__":
    cap = cv2.VideoCapture("videos_labeled/1.hevc")
    while cap.isOpened():
        ret, frame = cap.read()
        if ret == True:
            process_frame(frame)
        else :
            break



