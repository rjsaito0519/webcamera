import matplotlib.pyplot as plt
import cv2
import numpy as np
import glob
from tqdm import tqdm

class AnalysisImage:
    def __init__(self, path):
        self.gray_img = cv2.cvtColor( cv2.imread(path) , cv2.COLOR_BGR2GRAY)[10:-10, 20:-20]
        self.binary   = None
        
        self.num       = 0
        self.area      = np.array([])
        self.arc_len   = np.array([])
        self.luminance = np.array([])
        self.rect      = []
        self.ratio     = np.array([])

    def analyze(self, threshold = 5):
        ret, self.binary = cv2.threshold(self.gray_img, threshold, 255, cv2.THRESH_BINARY)
        contours, hierarchy = cv2.findContours(self.binary, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)

        self.num = len( contours )
        tmp_area         = []
        area_append      = tmp_area.append
        tmp_arc_len      = []
        arc_len_append   = tmp_arc_len.append
        tmp_luminance    = []
        luminance_append = tmp_luminance.append
        rect_append      = self.rect.append
        tmp_ratio        = []
        ratio_append     = tmp_ratio.append

        for i, contour in enumerate(tqdm(contours)):
            area_append( cv2.contourArea(contour) )
            arc_len_append( cv2.arcLength(contour, True) )
            mask = np.zeros_like(self.gray_img)
            cv2.fillPoly(mask, contour, 255)
            luminance_append( np.sum( (self.gray_img * mask).ravel() ) )
            tmp_rect = cv2.minAreaRect(contour)
            rect_append( tmp_rect )
            ratio_append( np.min(tmp_rect[1]) / np.max(tmp_rect[1]) )

        self.area      = np.array(tmp_area)
        self.arc_len   = np.array(tmp_arc_len)
        self.luminance = np.array(tmp_luminance)
        self.ratio     = np.array(tmp_ratio)

    def rect_plot(self, i):
        
        img_disp = cv2.cvtColor(self.gray_img, cv2.COLOR_GRAY2BGR)
        if i == -1:
            for j in range(self.num):
                box = cv2.boxPoints(self.rect[j])
                box = np.intp(box)
                cv2.drawContours(img_disp, [box], 0, (0,255,255))
        else:
            box = cv2.boxPoints(self.rect[i])
            box = np.intp(box)
            cv2.drawContours(img_disp, [box], 0, (0,255,255))

        plt.imshow(img_disp)
        plt.show()        
        
# a = glob.glob("./*.tiff")
# Sr = AnalysisImage( a[1] )
# Sr.analyze(10)
# Sr.rect_plot(500)

img = cv2.imread("152Eu_240.tiff")
img2 = cv2.imread("batch0-i49.tiff")

print( np.all( img == img2 ) )

# def rotate(data, theta):
#     theta = np.deg2rad(theta)
#     rot = np.array([
#         [np.cos(theta), -1*np.sin(theta)],
#         [np.sin(theta), np.cos(theta)]
#     ])
#     return np.dot( rot, data )

# a = glob.glob("./*.jpg")
# b = np.array([])

# print(b)
# img = cv2.imread(a[0])
# gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
# gray = gray[10:-10,20:-20]

# plt.imshow(gray)
# plt.show()

# #sets any pixels of intensity <5 into 0
# ret, thresh = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY)

# plt.imshow(thresh)
# plt.show()

# #finds contours
# contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)

# mask = np.zeros_like(gray)
# # Loop through each contour and compute its length, area, and total luminance
# data1 = []
# data2 = []

# for i, contour in enumerate(contours):

#     # # Compute the length of the contour
#     # detection['length'] = cv2.arcLength(contour, True)

#     # # Compute the area of the contour
#     # detection['area'] = cv2.contourArea(contour)
    
#     data1.append([ cv2.arcLength(contour, True), cv2.contourArea(contour) ])

#     # Compute the total luminance of the pixels within the contour
    
#     cv2.drawContours(gray, contours, i, color=255, thickness=1)
#     mask = np.zeros_like(gray)
#     cv2.fillPoly(mask, contours[i], 255)
    
#     rect = cv2.minAreaRect(contour)
#     box = cv2.boxPoints(rect)
#     print(rect, box)
#     box = np.intp(box)
    
#     cen = np.array([
#         [rect[0][0]],
#         [rect[0][1]]
#     ])
#     wh = rect[1]
#     theta = rect[2]
    
#     print( rotate( np.array([ [wh[0]/2], [wh[1]/2] ]), theta ) + cen )
#     data2.append(np.min(rect[1])/np.max(rect[1]))

#     # input()

#     # plt.imshow(thresh)
#     # plt.plot( box[:, 0], box[:, 1] )
#     # # for i in range(len(box)):
#     # #     plt.plot( box[i][0] )
#     # plt.show()

#     # aaaaaaaaaaa = input()
#     # detection['luminance'] = cv2.sumElems(grey * (mask > 0))[0]

#     # Print the results
#     #TURN OFF IF REPEATING A LOT
#     #print(f"Contour {i+1}: length={detection['length']:.2f}, area={detection['area']:.2f}, total luminance={detection['luminance']:.2f}")

#     # Check if the length is 5 or above
#     #currently set to 0. To change, change the value after ">="
#     # if detection['length'] >= 0.01 and detection['area'] >= 0.01 and detection['luminance'] >= 0.01:

#     #     # Store the results in the lists
#     #     detection['lengths'].append(detection['length'])
#     #     detection['areas'].append(detection['area'])
#     #     detection['luminances'].append(detection['luminance'])
#     #     detection['Tcontour'] += 1
# # plt.imshow(mask)
# # plt.show()

# data1 = np.array(data1)

# plt.hist(data2, bins = 10)
# plt.show()

# plt.hist(data1[ data1[:, 0] < 200 ][:, 0], bins = 100)
# plt.yscale("log")
# plt.show()

# plt.hist(data1[ data1[:, 1] < 200 ][:, 1], bins = 100)
# plt.yscale("log")
# plt.show()