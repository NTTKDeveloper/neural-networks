import torch
import cv2
import data


img = cv2.imread("./0.png", cv2.IMREAD_GRAYSCALE)
img = img.flatten()
inputs = img/255
#inputs, result, count = data.loadata(0, 999)

inputs = torch.tensor(inputs, dtype=torch.float32)

#Load model
model = torch.load("/home/tuankhanh/Desktop/neural-networks/save/core.pt")

y_pred = model(inputs)

print(y_pred)

print(torch.argmax(y_pred))