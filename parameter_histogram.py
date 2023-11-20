# original class import
import webcam

# Standard imports
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.ticker import ScalarFormatter
import cv2
import numpy as np
import glob
import pandas as pd
from tqdm import tqdm

plt.rcParams['font.family'] = 'Times New Roman' #全体のフォントを設定
plt.rcParams['mathtext.fontset'] = 'stix'
plt.rcParams['font.size'] = 22
plt.rcParams['axes.linewidth'] = 1.0# 軸の線幅edge linewidth。囲みの太さ
plt.rcParams['axes.grid'] = True
plt.rcParams['figure.subplot.left'] = 0.13
plt.rcParams['figure.subplot.right'] = 0.95
plt.rcParams['figure.subplot.top'] = 0.95
plt.rcParams['figure.subplot.bottom'] = 0.13
plt.rcParams["xtick.direction"] = "in"               #x軸の目盛線を内向きへ
plt.rcParams["ytick.direction"] = "in"               #y軸の目盛線を内向きへ
plt.rcParams["xtick.minor.visible"] = True           #x軸補助目盛りの追加
plt.rcParams["ytick.minor.visible"] = True           #y軸補助目盛りの追加
plt.rcParams["xtick.major.size"] = 10                #x軸主目盛り線の長さ
plt.rcParams["ytick.major.size"] = 10                #y軸主目盛り線の長さ
plt.rcParams["xtick.minor.size"] = 5                 #x軸補助目盛り線の長さ
plt.rcParams["ytick.minor.size"] = 5                 #y軸補助目盛り線の長さ

test_slice = slice( 0, 6 )     
img_threshold = 15

Na_pathlist = glob.glob("./data/22Na/*.tiff")
Na = webcam.AnalysisImage( Na_pathlist[test_slice], webcam.Na_slice )
Na.analyze(img_threshold)

Co_pathlist = glob.glob("./data/60Co/*.tiff")
Co = webcam.AnalysisImage( Co_pathlist[test_slice], webcam.Co_slice )
Co.analyze(img_threshold)

Sr_pathlist = glob.glob("./data/90Sr/*.tiff")
Sr = webcam.AnalysisImage( Sr_pathlist[test_slice], webcam.Sr_slice )
Sr.analyze(img_threshold)

Ba_pathlist = glob.glob("./data/133Ba/*.tiff")
Ba = webcam.AnalysisImage( Ba_pathlist[test_slice], webcam.Ba_slice )
Ba.analyze(img_threshold)

Eu_pathlist = glob.glob("./data/152Eu/*.tiff")
Eu = webcam.AnalysisImage( Eu_pathlist[test_slice], webcam.Eu_slice )
Eu.analyze(img_threshold)

Am_pathlist = glob.glob("./data/241Am/*.tiff")
Am = webcam.AnalysisImage( Am_pathlist[test_slice], webcam.Am_slice )
Am.analyze(img_threshold)

Monazu_pathlist = glob.glob("./data/Monazu/*.tiff")
Monazu = webcam.AnalysisImage( Monazu_pathlist[test_slice], webcam.Monazu_slice )
Monazu.analyze(img_threshold)

fig = plt.figure(figsize= (8, 8))
ax = fig.add_subplot(1, 1, 1)
ax.hist( Na.luminance, bins = 50, range = (0, 8000), alpha = 0.3, density=1., label = "22Na", color = "C0" )
ax.hist( Sr.luminance, bins = 50, range = (0, 8000), alpha = 0.3, density=1., label = "90Sr", color = "C1" )
ax.hist( Co.luminance, bins = 50, range = (0, 8000), alpha = 0.3, density=1., label = "60Co", color = "C2" )
ax.hist( Ba.luminance, bins = 50, range = (0, 8000), alpha = 0.3, density=1., label = "133Ba", color = "C3" )
ax.hist( Eu.luminance, bins = 50, range = (0, 8000), alpha = 0.3, density=1., label = "152Eu", color = "C4" )
ax.hist( Am.luminance, bins = 50, range = (0, 8000), alpha = 0.3, density=1., label = "241Am", color = "C5" )
ax.set_xlabel("Luminance")
ax.set_xlim( -10, 5100 )
ax.yaxis.set_major_formatter(ScalarFormatter(useMathText=True))
ax.ticklabel_format(style="sci",  axis="y",scilimits=(0,0))
plt.legend()
plt.savefig("img/luminance_all.png", dpi=600, transparent=True)
plt.show()

fig = plt.figure(figsize= (8, 8))
ax = fig.add_subplot(1, 1, 1)
ax.hist( Co.luminance, bins = 50, range = (0, 8000), alpha = 0.3, density=1., label = "60Co", color = "C2" )
ax.hist( Ba.luminance, bins = 50, range = (0, 8000), alpha = 0.3, density=1., label = "133Ba", color = "C3" )
ax.hist( Eu.luminance, bins = 50, range = (0, 8000), alpha = 0.3, density=1., label = "152Eu", color = "C4" )
ax.set_xlabel("Luminance")
ax.set_xlim( -10, 5100 )
ax.yaxis.set_major_formatter(ScalarFormatter(useMathText=True))
ax.ticklabel_format(style="sci",  axis="y",scilimits=(0,0))
plt.legend()
plt.savefig("img/luminance_gammma.png", dpi=600, transparent=True)
plt.show()

fig = plt.figure(figsize= (8, 8))
ax = fig.add_subplot(1, 1, 1)
ax.hist( Na.luminance, bins = 50, range = (0, 8000), alpha = 0.3, density=1., label = "22Na", color = "C0" )
ax.hist( Sr.luminance, bins = 50, range = (0, 8000), alpha = 0.3, density=1., label = "90Sr", color = "C1" )
ax.set_xlabel("Luminance")
ax.set_xlim( -10, 5100 )
ax.yaxis.set_major_formatter(ScalarFormatter(useMathText=True))
ax.ticklabel_format(style="sci",  axis="y",scilimits=(0,0))
plt.legend()
plt.savefig("img/luminance_beta.png", dpi=600, transparent=True)
plt.show()

fig = plt.figure(figsize= (8, 8))
ax = fig.add_subplot(1, 1, 1)
ax.hist( Na.area, bins = 50, range = (0, 40), alpha = 0.3, density=1., label = "22Na", color = "C0" )
ax.hist( Sr.area, bins = 50, range = (0, 40), alpha = 0.3, density=1., label = "90Sr", color = "C1" )
ax.hist( Co.area, bins = 50, range = (0, 40), alpha = 0.3, density=1., label = "60Co", color = "C2" )
ax.hist( Ba.area, bins = 50, range = (0, 40), alpha = 0.3, density=1., label = "133Ba", color = "C3" )
ax.hist( Eu.area, bins = 50, range = (0, 40), alpha = 0.3, density=1., label = "152Eu", color = "C4" )
ax.hist( Am.area, bins = 50, range = (0, 40), alpha = 0.3, density=1., label = "241Am", color = "C5" )
ax.set_xlabel("Area")
ax.set_xlim( -1, 21 )
ax.yaxis.set_major_formatter(ScalarFormatter(useMathText=True))
ax.ticklabel_format(style="sci",  axis="y",scilimits=(0,0))
plt.legend()
plt.savefig("img/area_all.png", dpi=600, transparent=True)
plt.show()

fig = plt.figure(figsize= (8, 8))
ax = fig.add_subplot(1, 1, 1)
ax.hist( Na.arc_len, bins = 50, range = (0, 40), alpha = 0.3, density=1., label = "22Na", color = "C0" )
ax.hist( Sr.arc_len, bins = 50, range = (0, 40), alpha = 0.3, density=1., label = "90Sr", color = "C1" )
ax.hist( Co.arc_len, bins = 50, range = (0, 40), alpha = 0.3, density=1., label = "60Co", color = "C2" )
ax.hist( Ba.arc_len, bins = 50, range = (0, 40), alpha = 0.3, density=1., label = "133Ba", color = "C3" )
ax.hist( Eu.arc_len, bins = 50, range = (0, 40), alpha = 0.3, density=1., label = "152Eu", color = "C4" )
ax.hist( Am.arc_len, bins = 50, range = (0, 40), alpha = 0.3, density=1., label = "241Am", color = "C5" )
ax.set_xlabel("Arc Length")
ax.set_xlim( -1, 31 )
ax.yaxis.set_major_formatter(ScalarFormatter(useMathText=True))
ax.ticklabel_format(style="sci",  axis="y",scilimits=(0,0))
plt.legend()
plt.savefig("img/arc_length__all.png", dpi=600, transparent=True)
plt.show()