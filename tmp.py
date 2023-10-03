import cv2, os
import numpy as np

files = os.listdir("v1/label/img")

for file in files:
    if 'png' in file:
        img = cv2.imread("v1/label/img/" + file)
        img = img/255
        img = np.array(img, np.uint8)
        cv2.imwrite("v1/label/img/" + file, img)
        print(f"{file} saved.")
    
    else:
        print(f"{file} skipped")