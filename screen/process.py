import cv2
import math
import numpy as np


def place_dot(img,x,y):
    cv2.rectangle(img,(x,y),(x+1,y+1),(255,0,0),1)

def lines(img,x,y):

    color = img[y-10][x]
    #j = y-10
    img2 = img.copy()
    dist = [0,0,0,0,0]
    for delta_x in range(-2,3):
        #i = x+delta_x
        j = y-5+abs(delta_x)
        i = x
        
        try:
            j -= 1 if (delta_x != -2 and delta_x != 2) else 0
            i = i+delta_x*5
            initial = sum(img[j][i])/3
            #print(initial)
            while sum(img[j][i])/3 > 150 and abs( sum(img[j][i])/3 - initial ) < 50 and abs(y-j) < 50 and abs(x-i) < 50:
                j -= 1 if (delta_x != -2 and delta_x != 2) else 0
                if delta_x == 0 or j % abs(delta_x) == 0:
                    i = i+ np.sign(delta_x)
            #print(sum(img[j][i])/3)
        except IndexError:
            pass

        #print(delta_x,x,i,img[j][i])
        
        cv2.line(img2,(x+delta_x*5,y-5+abs(delta_x)),(i,j),(255,0,0),1)
        dist[delta_x+2] = math.hypot(i-(x+delta_x*5),j-(y-5+abs(delta_x)))

    dist2 = [0,0,0,0,0]
    for delta_x in range(-1,2):
        #i = x+delta_x
        j = y+35
        i = x
        
        try:
            j += 1
            i = i+delta_x
            initial = sum(img[j][i])/3
            #print(initial)
            while sum(img[j][i])/3 > 150 and abs( sum(img[j][i])/3 - initial ) < 50 and abs(y-j+35) < 50:
                j += 1
                if delta_x == 0 or j % abs(delta_x) == 0:
                    i = i+ np.sign(delta_x)
            #print(sum(img[j][i])/3)
        except IndexError:
            pass

        #print(delta_x,x,i,img[j][i])
        
        cv2.line(img2,(x,y+35),(i,j),(255,0,0),1)
        dist2[delta_x+2] = math.hypot(i-x,j-(y+35))

    #print(dist,dist2,dist3,dist4)
    #cv2.imshow('hi2',img)
    return img2, dist+dist2