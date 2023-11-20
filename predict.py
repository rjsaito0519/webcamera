# original class import
import webcam

# Standard imports
import matplotlib.pyplot as plt
import matplotlib as mpl
import cv2
import numpy as np
import glob
import pandas as pd
from tqdm import tqdm

# Keras/TensorFlow imports
import tensorflow as tf

plt.rcParams['font.family'] = 'Times New Roman' #全体のフォントを設定
plt.rcParams['mathtext.fontset'] = 'stix'
plt.rcParams['font.size'] = 22
plt.rcParams['axes.linewidth'] = 1.0# 軸の線幅edge linewidth。囲みの太さ
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

def predict_plot(model, dataset, title = ""):
    pred_data = model.predict(dataset)
    data = []
    for line in pred_data:
        data.append( np.argmax(line) )
    fig = plt.figure(figsize= (8, 8))
    plt.hist(data, bins = 3)
    plt.title(title)
    plt.savefig("img/{}.png".format(title), dpi=600, transparent=True)
    plt.show()

test_slice = slice( -6, None )     
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


loaded_model = tf.keras.models.load_model('BaSrAm_model.h5')

dataset = np.array([
    Na.area,
    Na.arc_len,
    Na.luminance
]).T
predict_plot(loaded_model, dataset, "Na")

dataset = np.array([
    Co.area,
    Co.arc_len,
    Co.luminance
]).T
predict_plot(loaded_model, dataset, "Co")

dataset = np.array([
    Sr.area,
    Sr.arc_len,
    Sr.luminance
]).T
predict_plot(loaded_model, dataset, "Sr")

dataset = np.array([
    Ba.area,
    Ba.arc_len,
    Ba.luminance
]).T
predict_plot(loaded_model, dataset, "Ba")

dataset = np.array([
    Eu.area,
    Eu.arc_len,
    Eu.luminance
]).T
predict_plot(loaded_model, dataset, "Eu")

dataset = np.array([
    Am.area,
    Am.arc_len,
    Am.luminance
]).T
predict_plot(loaded_model, dataset, "Am")

dataset = np.array([
    Monazu.area,
    Monazu.arc_len,
    Monazu.luminance
]).T

predict_plot(loaded_model, dataset, "Monazu")