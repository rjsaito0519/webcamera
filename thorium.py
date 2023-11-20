import numpy as np
import matplotlib.pyplot as plt
import lmfit as lf
import lmfit.models as lfm

plt.rcParams['font.family'] = 'Times New Roman' #全体のフォントを設定
plt.rcParams['mathtext.fontset'] = 'stix'
plt.rcParams['font.size'] = 22
plt.rcParams['axes.linewidth'] = 1.0# 軸の線幅edge linewidth。囲みの太さ
plt.rcParams['axes.grid'] = True
plt.rcParams['figure.subplot.left'] = 0.13
plt.rcParams['figure.subplot.right'] = 0.97
plt.rcParams['figure.subplot.top'] = 0.97
plt.rcParams['figure.subplot.bottom'] = 0.13
plt.rcParams["xtick.direction"] = "in"               #x軸の目盛線を内向きへ
plt.rcParams["ytick.direction"] = "in"               #y軸の目盛線を内向きへ
plt.rcParams["xtick.minor.visible"] = True           #x軸補助目盛りの追加
plt.rcParams["ytick.minor.visible"] = True           #y軸補助目盛りの追加
plt.rcParams["xtick.major.size"] = 10                #x軸主目盛り線の長さ
plt.rcParams["ytick.major.size"] = 10                #y軸主目盛り線の長さ
plt.rcParams["xtick.minor.size"] = 5                 #x軸補助目盛り線の長さ
plt.rcParams["ytick.minor.size"] = 5                 #y軸補助目盛り線の長さ


def data2hist(data, bin_num = 100, range = None):
    try:
        hist_data, bin = np.histogram(data, bins = bin_num, range = range)
    except:
        hist_data, bin = np.histogram(data, bins = bin_num)
    bin_width = (bin[-1]-bin[0])/bin_num
    x = bin[:-1] + bin_width/2 * np.ones_like(bin[:-1])
    return x, hist_data

def exp_fit(data, bin_num = 100, range = None):
    x, y = data2hist( data, bin_num = 100, range = None )
    model = lfm.ExponentialModel()
    params = model.guess(x = x, data = y)
   
    result = model.fit(x=x, data=y, params=params, method='leastsq')
    print(result.fit_report())

    fit_x = np.linspace(np.min(x), np.max(x), 10000)
    fit_y = result.eval_components(x=fit_x)["exponential"]

    fig = plt.figure(figsize = (8, 8))
    ax = fig.add_subplot(1,1,1)
    ax.plot( x, y, "o", color = "w", markeredgecolor = "k")
    ax.plot( fit_x, fit_y )
    ax.set_xlabel("time [s]")
    ax.set_ylabel("count")
    plt.show()

class RI:
    def __init__(self, lifetime, mode):
        self.lifetime = lifetime
        self.mode = mode

    def decay(self, size = 1000):
        rand = np.random.exponential(scale=self.lifetime, size=size)
        return rand


Rn = RI(55.6, { "alpha": 100 })
a = Rn.decay()

exp_fit(a)

plt.hist(a)
plt.show()