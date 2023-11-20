# Standard imports
import matplotlib.pyplot as plt
import matplotlib as mpl
import cv2
import numpy as np
import glob
import pandas as pd
from tqdm import tqdm

# 写真データのオフセット分を取り除く用のパラメタ
Na_slice = ( slice( 70, -71 ), slice(  2,  -3 ) )
Co_slice = ( slice( 70, -71 ), slice(  2,  -3 ) )
Sr_slice = ( slice( None ),    slice( None ) )
Ba_slice = ( slice( 44, -45 ), slice(  2,  -3 ) )
Eu_slice = ( slice( 43, -44 ), slice(  2,  -3 ) )
Am_slice = ( slice(  2,  -3 ), slice( 18, -19 ) )
Monazu_slice = ( slice( 2, -3 ), slice( 47, -48 ) )

class AnalysisImage:
    def __init__(self, pathlist, source_slice):
        self.pathlist = pathlist
        self.source_slice = source_slice
        self.threshold = 5

        self.num       = []
        self.area      = np.array([])
        self.arc_len   = np.array([])
        self.luminance = np.array([])
        self.ratio     = np.array([])

    def initialize(self):
        self.num       = []
        self.area      = np.array([])
        self.arc_len   = np.array([])
        self.luminance = np.array([])
        self.ratio     = np.array([])

    def load_data(self, path):
        gray_img = cv2.cvtColor( cv2.imread(path) , cv2.COLOR_BGR2GRAY)[ self.source_slice[0], self.source_slice[1] ]
        return gray_img

    def analyze(self, threshold = 5):
        self.threshold = threshold
        for path in tqdm(self.pathlist):
            gray_img = self.load_data( path )
            ret, binary = cv2.threshold(gray_img, self.threshold, 255, cv2.THRESH_BINARY)
            contours, hierarchy = cv2.findContours(binary, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)

            self.num.append( len(contours) )
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

    def get_img_event_num(self, n):
        img_num = 0
        buff = 0
        for i in range(len(self.num)):
            buff += self.num[i]
            if n <= buff:
                img_num = i
                break
        event_num = n - ( buff - self.num[img_num] )

        return img_num, event_num

    def rect_plot(self, img_num, save_name = "rect_plot", isSave = False):
        gray_img = self.load_data( self.pathlist[img_num] )
        ret, binary = cv2.threshold(gray_img, self.threshold, 255, cv2.THRESH_BINARY)
        contours, hierarchy = cv2.findContours(binary, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
        img_disp = cv2.cvtColor(gray_img, cv2.COLOR_GRAY2BGR)
        for i, contour in enumerate(contours):
            rect = cv2.minAreaRect(contour)
            box = cv2.boxPoints(rect)
            box = np.intp(box)
            cv2.drawContours(img_disp, [box], 0, (255,255,0))
        fig1 = plt.figure( figsize=( 8, 6 ) )
        ax1 = fig1.add_subplot(111)
        ax1.imshow(gray_img)
        fig2 = plt.figure( figsize=( 8, 6 ) )
        ax2 = fig2.add_subplot(111)
        ax2.imshow(binary)
        fig3 = plt.figure( figsize=( 8, 6 ) )
        ax3 = fig3.add_subplot(111)
        ax3.imshow(img_disp)
        if isSave:
            fig1.savefig("img/{}_gray.png".format(save_name), dpi=600, transparent=True)
            fig2.savefig("img/{}_binary.png".format(save_name), dpi=600, transparent=True)
            fig3.savefig("img/{}}_color.png".format(save_name), dpi=600, transparent=True)
        plt.show()

    def event_plot(self, img_num, event_num, save_name = "event_plot", isSave = False):
        gray_img = self.load_data( self.pathlist[img_num] )
        ret, binary = cv2.threshold(gray_img, self.threshold, 255, cv2.THRESH_BINARY)
        contours, hierarchy = cv2.findContours(binary, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
        img_disp = cv2.cvtColor(gray_img, cv2.COLOR_GRAY2BGR)

        rect = cv2.minAreaRect(contours[event_num])
        box = cv2.boxPoints(rect)
        box = np.intp(box)
        # cv2.drawContours(img_disp, [box], 0, (0,255,255))
        fig = plt.figure(figsize=(8, 8))
        ax = fig.add_subplot(111)
        ax.imshow(img_disp)
        ax.set_xlim(np.min( box[:, 0] ) - 5, np.max( box[:, 0] ) + 5)
        ax.set_ylim(np.min( box[:, 1] ) - 5, np.max( box[:, 1] ) + 5)
        if isSave:
            plt.savefig("img/{}.png".format(save_name), dpi=600, transparent=True)
        plt.show()
