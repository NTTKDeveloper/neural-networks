import cv2
import numpy as np

#Add image 
img_0 = cv2.imread("0.png", cv2.IMREAD_GRAYSCALE)

print(img_0.shape)

#So luong pixel trang
number_white_pixel = np.sum(img_0 == 255)
print(number_white_pixel)

number_black_pixel = np.sum(img_0 == 0)
print(number_black_pixel)
print("Tong cong: 784")
print(number_white_pixel + number_black_pixel)

#Cat theo chieu kim dong ho
img_1 = img_0[0:14,0:14]
#img_2 = img_0[0:14,0:14]
#img_3 = img_0[0:14,0:14]
#img_4 = img_0[0:14,0:14]
print(img_1.shape)

pixel_whiteimg1 = np.sum(img_1 == 255)
pixel_blackimg1 = np.sum(img_1 == 0)
print(pixel_whiteimg1)
print(pixel_blackimg1)


