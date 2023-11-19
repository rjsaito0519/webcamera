# Standard imports
import matplotlib.pyplot as plt
import matplotlib as mpl
import cv2
import numpy as np
import glob
import pandas as pd
from tqdm import tqdm

import tensorflow as tf


Na_slice = ( slice( 70, -71 ), slice(  2,  -3 ) )
Co_slice = ( slice( 70, -71 ), slice(  2,  -3 ) )
Sr_slice = ( slice( None ),    slice( None ) )
Ba_slice = ( slice( 44, -45 ), slice(  2,  -3 ) )
Eu_slice = ( slice( 43, -44 ), slice(  2,  -3 ) )
Am_slice = ( slice(  2,  -3 ), slice( 18, -19 ) )
Monazu_slice = ( slice( 47, -48 ), slice( 2, -3 ) )

class AnalysisImage:
    def __init__(self, pathlist, source_slice):
        self.pathlist = pathlist
        self.source_slice = source_slice

        self.num       = 0
        self.area      = np.array([])
        self.arc_len   = np.array([])
        self.luminance = np.array([])
        self.ratio     = np.array([])

    def initialize(self):
        self.num       = 0
        self.area      = np.array([])
        self.arc_len   = np.array([])
        self.luminance = np.array([])
        self.ratio     = np.array([])

    def load_data(self, path):
        gray_img = cv2.cvtColor( cv2.imread(path) , cv2.COLOR_BGR2GRAY)[ self.source_slice[0], self.source_slice[1] ]
        return gray_img

    def analyze(self, threshold = 5):
        
        for path in tqdm(self.pathlist):
            gray_img = self.load_data( path )
            ret, binary = cv2.threshold(gray_img, threshold, 255, cv2.THRESH_BINARY)
            contours, hierarchy = cv2.findContours(binary, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)

            self.num += len( contours )
            tmp_area         = []
            area_append      = tmp_area.append
            tmp_arc_len      = []
            arc_len_append   = tmp_arc_len.append
            tmp_luminance    = []
            luminance_append = tmp_luminance.append
            tmp_ratio        = []
            ratio_append     = tmp_ratio.append

            for i, contour in enumerate(contours):
                area_append( cv2.contourArea(contour) )
                arc_len_append( cv2.arcLength(contour, True) )
                mask = np.zeros_like(gray_img)
                cv2.fillPoly(mask, contour, 255)
                luminance_append( np.sum( (gray_img * mask).ravel() ) )
                tmp_rect = cv2.minAreaRect(contour)
                if np.max(tmp_rect[1]) != 0.:
                    ratio_append( np.min(tmp_rect[1]) / np.max(tmp_rect[1]) )
                else:
                    ratio_append( -1 )

            self.area      = np.append(self.area, tmp_area)
            self.arc_len   = np.append(self.arc_len, tmp_arc_len)
            self.luminance = np.append(self.luminance, tmp_luminance)
            self.ratio     = np.append(self.ratio, tmp_ratio)

    # def rect_plot(self, i):
        
    #     img_disp = cv2.cvtColor(self.gray_img, cv2.COLOR_GRAY2BGR)
    #     if i == -1:
    #         for j in range(self.num):
    #             box = cv2.boxPoints(self.rect[j])
    #             box = np.intp(box)
    #             cv2.drawContours(img_disp, [box], 0, (0,255,255))
    #     else:
    #         box = cv2.boxPoints(self.rect[i])
    #         box = np.intp(box)
    #         cv2.drawContours(img_disp, [box], 0, (0,255,255))

    #     plt.imshow(img_disp)
    #     plt.show()        

test_slice = slice( 0, 1 )     
img_threshold = 15
# Na_pathlist = glob.glob("./data/22Na/*.tiff")
# Na = AnalysisImage( Na_pathlist[test_slice], Na_slice )
# Na.analyze(img_threshold)

Co_pathlist = glob.glob("./data/60Co/*.tiff")
Co = AnalysisImage( Co_pathlist[test_slice], Co_slice )
Co.analyze(img_threshold)

# Sr_pathlist = glob.glob("./data/90Sr/*.tiff")
# Sr = AnalysisImage( Sr_pathlist[test_slice], Sr_slice )
# Sr.analyze(img_threshold)

Ba_pathlist = glob.glob("./data/133Ba/*.tiff")
Ba = AnalysisImage( Ba_pathlist[test_slice], Ba_slice )
Ba.analyze(img_threshold)

Eu_pathlist = glob.glob("./data/152Eu/*.tiff")
Eu = AnalysisImage( Eu_pathlist[test_slice], Eu_slice )
Eu.analyze(img_threshold)

Am_pathlist = glob.glob("./data/241Am/*.tiff")
Am = AnalysisImage( Am_pathlist[test_slice], Am_slice )
Am.analyze(img_threshold)

Monazu_pathlist = glob.glob("./data/Monazu/*.tiff")
Monazu = AnalysisImage( Monazu_pathlist[test_slice], Monazu_slice )
Monazu.analyze(img_threshold)


loaded_model = tf.keras.models.load_model('test_model.h5')

dataset4 = np.array([
    Monazu.area,
    Monazu.arc_len,
    Monazu.luminance
]).T

# dataset4 = np.array([
#     Co.area,
#     Co.arc_len,
#     Co.luminance
# ]).T

# dataset4 = np.array([
#     Eu.area,
#     Eu.arc_len,
#     Eu.luminance
# ]).T

# dataset4 = np.array([
#     Am.area,
#     Am.arc_len,
#     Am.luminance
# ]).T

# dataset4 = np.array([
#     Ba.area,
#     Ba.arc_len,
#     Ba.luminance
# ]).T


pred_data = loaded_model.predict(dataset4)
print(pred_data)

data = []

for line in pred_data:
    data.append( np.argmax(line) )

print(data)
plt.hist(data, bins = 3)
plt.show()