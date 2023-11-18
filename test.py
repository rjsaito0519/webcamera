import matplotlib.pyplot as plt
import cv2
import numpy as np
import glob

a = glob.glob("./*.tiff")

img = cv2.imread(a[0])
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
gray = gray[10:-10,20:-20]

# plt.imshow(gray)
# plt.show()

#sets any pixels of intensity <5 into 0
ret, thresh = cv2.threshold(gray, 5, 255, cv2.THRESH_BINARY)

# plt.imshow(thresh)
# plt.show()

#finds contours
contours, hierarchy = cv2.findContours(thresh, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
mask = np.zeros_like(gray)
# Loop through each contour and compute its length, area, and total luminance
data1 = []
for i, contour in enumerate(contours):

    # # Compute the length of the contour
    # detection['length'] = cv2.arcLength(contour, True)

    # # Compute the area of the contour
    # detection['area'] = cv2.contourArea(contour)
    
    data1.append([ cv2.arcLength(contour, True), cv2.contourArea(contour) ])

    # Compute the total luminance of the pixels within the contour
    
    cv2.drawContours(gray, contours, i, color=255, thickness=1)
    cv2.drawContours(mask, contours, i, color=255, thickness=1)
    

    # aaaaaaaaaaa = input()
    # detection['luminance'] = cv2.sumElems(grey * (mask > 0))[0]

    # Print the results
    #TURN OFF IF REPEATING A LOT
    #print(f"Contour {i+1}: length={detection['length']:.2f}, area={detection['area']:.2f}, total luminance={detection['luminance']:.2f}")

    # Check if the length is 5 or above
    #currently set to 0. To change, change the value after ">="
    # if detection['length'] >= 0.01 and detection['area'] >= 0.01 and detection['luminance'] >= 0.01:

    #     # Store the results in the lists
    #     detection['lengths'].append(detection['length'])
    #     detection['areas'].append(detection['area'])
    #     detection['luminances'].append(detection['luminance'])
    #     detection['Tcontour'] += 1
# plt.imshow(mask)
# plt.show()

data1 = np.array(data1)

plt.hist(data1[:, 0], bins = 100, alpha = 0.5, density=1)
plt.show()

plt.hist(data1[:, 1], bins = 100, alpha = 0.5, density=1)
plt.show()