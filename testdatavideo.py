import cv2
import torch
import time 
import pyautogui
import numpy as np

cap = cv2.VideoCapture("Data_1.mp4")
fps = 1
wait_time = 1000/fps

#Load model 
model = torch.load("/home/tuankhanh/Desktop/neural-networks/save/core.pt")


def process_speed_1(img, y0, y1, x0, x1):
    # Create ROI km/h
    img_1= img[y0:y1, x0:x1]
    # Gray mode
    img_1 = cv2.cvtColor(img_1, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(img_1, (3,3), 0)
    thresh = cv2.threshold(blur, 200, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
    return thresh

while True:
    pre_time = time.time()
    ret,img = cap.read()

    cv2.imshow("Video Original", img)

    img = cv2.resize(img, (1280,720))


    img_1 = process_speed_1(img, 657, 688, 1111, 1134)
    img_1 = cv2.resize(img_1, (28,28))

    img_2 = process_speed_1(img, 657, 688, 1129, 1148)
    img_2 = cv2.resize(img_2, (28,28))

    img_3 = process_speed_1(img, 657, 688, 1148, 1166)
    img_3 = cv2.resize(img_3, (28,28))

    cv2.imshow("Image_1", img_2)

 #   print(img_1.shape)
    #Bien doi data
    img_2 = img_2.flatten()
#    print(img_1.shape)
    inputs = img_2/255
    inputs = torch.tensor(inputs, dtype=torch.float32)
    print(inputs.shape)

    result = model(inputs)
    print(result)
    result_index = torch.argmax(result)

    print(result_index)


    delta_time = (time.time() - pre_time)*1000
    delay_time = wait_time - delta_time
    if delta_time > wait_time:
        delay_time = 1
    else:
        delay_time = wait_time - delta_time
    # Nhấn q để thoát
    if cv2.waitKey(int(wait_time)) & 0xFF == ord('q'):
        cv2.destroyAllWindows()
        break

