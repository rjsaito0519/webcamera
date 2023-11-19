# Standard imports
import matplotlib.pyplot as plt
import matplotlib as mpl
import cv2
import numpy as np
import glob
import pandas as pd
from tqdm import tqdm


# Scikit-learn imports
from sklearn.model_selection import train_test_split

# Keras/TensorFlow imports
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.optimizers import Adam

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
Na_pathlist = glob.glob("./data/22Na/*.tiff")
Na = AnalysisImage( Na_pathlist[test_slice], Na_slice )
Na.analyze(img_threshold)

# Co_pathlist = glob.glob("./data/60Co/*.tiff")
# Co = AnalysisImage( Co_pathlist[test_slice], Co_slice )
# Co.analyze(img_threshold)

Sr_pathlist = glob.glob("./data/90Sr/*.tiff")
Sr = AnalysisImage( Sr_pathlist[test_slice], Sr_slice )
Sr.analyze(img_threshold)

# Ba_pathlist = glob.glob("./data/133Ba/*.tiff")
# Ba = AnalysisImage( Ba_pathlist[test_slice], Ba_slice )
# Ba.analyze(img_threshold)

# Eu_pathlist = glob.glob("./data/152Eu/*.tiff")
# Eu = AnalysisImage( Eu_pathlist[test_slice], Eu_slice )
# Eu.analyze(img_threshold)

Am_pathlist = glob.glob("./data/241Am/*.tiff")
Am = AnalysisImage( Am_pathlist[test_slice], Am_slice )
Am.analyze(img_threshold)

Monazu_pathlist = glob.glob("./data/Monazu/*.tiff")
Monazu = AnalysisImage( Monazu_pathlist[test_slice], Monazu_slice )
Monazu.analyze(img_threshold)

# fig = plt.figure(figsize= (8, 8))
# ax = fig.add_subplot(1, 1, 1)
# ax.hist( Na.luminance, bins = 50, alpha = 0.5, density=1., label = "22Na" )
# ax.hist( Co.luminance, bins = 50, alpha = 0.5, density=1., label = "60Co" )
# ax.hist( Sr.luminance, bins = 50, alpha = 0.5, density=1., label = "90Sr" )
# ax.hist( Ba.luminance, bins = 50, alpha = 0.5, density=1., label = "133Ba" )
# ax.hist( Eu.luminance, bins = 50, alpha = 0.5, density=1., label = "152Eu" )
# # ax.hist( Am.luminance, bins = 100, alpha = 0.5, density=1., label = "241Am" )
# plt.legend()
# plt.show()


# fig = plt.figure(figsize= (8, 8))
# ax = fig.add_subplot(1, 1, 1)
# ax.hist( Na.ratio, bins = 30, alpha = 0.5, density=1., label = "22Na" )
# ax.hist( Co.ratio, bins = 30, alpha = 0.5, density=1., label = "60Co" )
# ax.hist( Sr.ratio, bins = 30, alpha = 0.5, density=1., label = "90Sr" )
# ax.hist( Ba.ratio, bins = 30, alpha = 0.5, density=1., label = "133Ba" )
# ax.hist( Eu.ratio, bins = 30, alpha = 0.5, density=1., label = "152Eu" )
# # ax.hist( Am.ratio, bins = 30, alpha = 0.5, density=1., label = "241Am" )
# plt.legend()
# plt.show()


# fig = plt.figure(figsize= (8, 8))
# ax = fig.add_subplot(1, 1, 1)
# ax.hist( Na.arc_len, bins = 30, alpha = 0.5, density=1., label = "22Na" )
# ax.hist( Co.arc_len, bins = 30, alpha = 0.5, density=1., label = "60Co" )
# ax.hist( Sr.arc_len, bins = 30, alpha = 0.5, density=1., label = "90Sr" )
# ax.hist( Ba.arc_len, bins = 30, alpha = 0.5, density=1., label = "133Ba" )
# ax.hist( Eu.arc_len, bins = 30, alpha = 0.5, density=1., label = "152Eu" )
# ax.hist( Am.arc_len, bins = 30, alpha = 0.5, density=1., label = "241Am" )
# plt.legend()
# plt.show()




# fig = plt.figure()
# ax1 = fig.add_subplot(2,2,1)
# ax2 = fig.add_subplot(2,2,2)
# ax3 = fig.add_subplot(2,2,3)
# ax4 = fig.add_subplot(2,2,4)

# H = ax1.hist2d(Na.luminance, Na.ratio, bins=40, norm=mpl.colors.LogNorm(), cmap="viridis", label = "22Na")
# fig.colorbar(H[3],ax=ax1)

# H = ax2.hist2d(Co.luminance, Co.ratio, bins=40, norm=mpl.colors.LogNorm(), cmap="viridis", label = "60Co")
# fig.colorbar(H[3],ax=ax2)

# H = ax3.hist2d(Sr.luminance, Sr.ratio, bins=40, norm=mpl.colors.LogNorm(), cmap="viridis", label = "90Sr")
# fig.colorbar(H[3],ax=ax3)

# H = ax3.hist2d(Am.luminance, Am.ratio, bins=40, norm=mpl.colors.LogNorm(), cmap="viridis", label = "241Am")
# fig.colorbar(H[3],ax=ax4)

# plt.show()

dataset1 = np.array([
    Na.area,
    Na.arc_len,
    Na.luminance
]).T
dataset2 = np.array([
    Sr.area,
    Sr.arc_len,
    Sr.luminance
]).T
dataset3 = np.array([
    Am.area,
    Am.arc_len,
    Am.luminance
]).T

ans1 = np.full_like( Na.area, 0 )
ans2 = np.full_like( Sr.area, 1 )
ans3 = np.full_like( Am.area, 2 )

dataset = np.vstack([dataset1, dataset2, dataset3])
ans = np.hstack([ans1, ans2, ans3])

x_train, x_valid, t_train, t_valid = train_test_split(dataset, ans, test_size=0.3)

# モデルの定義
model = Sequential([
    Dense(units=10, activation='relu', input_dim=3),  # ノード数が3の層を追加。活性化関数はシグモイド関数。
    Dense(units=10, activation='relu'),  # ノード数が3の層を追加。活性化関数はシグモイド関数。
    Dense(units=3, activation='softmax')  # ノード数が1の層を追加。活性化関数はシグモイド関数。
])

es = EarlyStopping(patience=100, verbose=0, mode='auto', restore_best_weights=True)
opt = Adam()
model.compile(loss='sparse_categorical_crossentropy', optimizer=opt, metrics=['accuracy'])

#  トレーニング
history = model.fit(
    x=x_train,
    y=t_train,
    validation_data=(x_valid, t_valid),
    batch_size=2**4,  # バッチサイズ。一回のステップで全てのデータを使うようにする。
    epochs=1000,  # 学習のステップ数
    verbose=0,  # 1とするとステップ毎に誤差関数の値などが表示される
    callbacks=es,  # ここでコールバックを指定します。
)

# モデルの保存
model.save(filepath='test_model.h5', save_format='h5')

# loss_train = history.history['loss']
# loss_valid = history.history['val_loss']
# # ロス関数の推移をプロット
# plt.plot(loss_train, label='loss (train)')
# plt.plot(loss_valid, label='loss (valid)')
# # plt.yscale("log")
# plt.show()

result_batchnorm = pd.DataFrame(history.history)

# 目的関数の値
result_batchnorm[['loss', 'val_loss']].plot()
plt.yscale("log")
plt.show()
result_batchnorm[['accuracy', 'val_accuracy']].plot()
plt.show()


dataset4 = np.array([
    Monazu.area,
    Monazu.arc_len,
    Monazu.luminance
]).T
pred_data = model.predict(dataset4)
print(pred_data)