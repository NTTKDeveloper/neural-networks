import os 
import cv2
import numpy as np

def loadata(label, count):
    #Xu li labal
    path = "./sample/" + str(label) + "/" + str(label) + "_"+ str(count) +".png"
    if label == 0:
        result = [1, 0, 0, 0, 0, 0, 0, 0, 0 ,0]
    elif label == 1:
        result = [0, 1, 0, 0, 0, 0, 0, 0, 0 ,0]
    elif label == 2:
        result = [0, 0, 1, 0, 0, 0, 0, 0, 0 ,0]
    elif label == 3:
        result = [0, 0, 0, 1, 0, 0, 0, 0, 0 ,0]
    elif label == 4:
        result = [0, 0, 0, 0, 1, 0, 0, 0, 0 ,0]
    elif label == 5:
        result = [0, 0, 0, 0, 0, 1, 0, 0, 0 ,0]
    elif label == 6:
        result = [0, 0, 0, 0, 0, 0, 1, 0, 0 ,0]
    elif label == 7:
        result = [0, 0, 0, 0, 0, 0, 0, 1, 0 ,0]
    elif label == 8:
        result = [0, 0, 0, 0, 0, 0, 0, 0, 1 ,0]
    elif label == 9:
        result = [0, 0, 0, 0, 0, 0, 0, 0, 0 ,1]
    #Load Image 
    #./sample/0/0_1.png
    img = cv2.imread(path)
    img = img.flatten()
    inputs = img/255
    return img, result, count

#label = 9 

#inputs, result = loadata(label, 10)
#print(inputs)
#print(result)

#img = cv2.imread(path)
#cv2.imshow("Display", img)
#cv2.waitKey()





