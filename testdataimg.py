import time 
import torch
import cv2
import data

start_time = time.time()

img = cv2.imread("./img_test/0.png", cv2.IMREAD_GRAYSCALE)
img = img.flatten()
inputs = img/255

inputs = torch.tensor(inputs, dtype=torch.float32)

#Load model
model = torch.load("./save/core.pt")

y_pred = model(inputs)

print(y_pred)

print(torch.argmax(y_pred))

endtime = time.time()

print("Runtime for program:{0}".format(endtime - start_time) )
