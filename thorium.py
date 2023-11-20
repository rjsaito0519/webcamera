import numpy as np
import matplotlib.pyplot as plt
import lmfit as lf
import lmfit.models as lfm
import statistics
import uncertainties

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
        # gamma: 0, beta: 1, alpha: 2
        self.mode = mode

    def decay(self, size = 1000):
        exp_rand = np.random.exponential(scale=self.lifetime, size=size)
        uni_rand = np.random.rand(size)*100
        decay_time = dict()
        ratio = 0.
        for key in self.mode.keys():
            decay_time[key] = exp_rand[ ( ratio <= uni_rand ) & ( uni_rand < ratio + self.mode[key] ) ]
            ratio += self.mode[key]
        return decay_time


def sim(N_init = 10**6, time_threshold = 20000):
    Rn220 = RI(55.6, { "2": 100. })
    t1 = Rn220.decay(N_init)

    Po216 = RI(0.145, {"2": 100.})
    t2 = Po216.decay( len(t1["2"]) )

    Pb212 = RI(10.64*3600, {"1": 100.})
    t3 = Pb212.decay( len(t2["2"]) )

    Bi212 = RI(60.55*60, {"1": 64.06, "2": 35.94})
    t4 = Bi212.decay( len(t3["1"]) )

    Po212 = RI(2.99*10**-7, {"2": 100.})
    t5_1 = Po212.decay( len(t4["1"]) )

    Tl208 = RI(3.083*60, {"1": 100.})
    t5_2 = Tl208.decay( len(t4["2"]) )

    alpha = 0
    beta = 0
    n_Po212 = len(t4["1"])
    for i in range(N_init):
        tmp_time = 0
        flag = False
        for j, t in enumerate([ t1["2"][i], t2["2"][i], t3["1"][i] ]):
            tmp_time = t
            if tmp_time < time_threshold:
                if j == 2:
                    beta += 1
                else:
                    alpha += 1
            else:
                flag = True
                break
        if flag:
            continue
        if i < n_Po212:
            for j, t in enumerate([ t4["1"][i], t5_1["2"][i] ]):
                tmp_time = t
                if tmp_time < time_threshold:
                    if j == 0:
                        beta += 1
                    else:
                        alpha += 1
                else:
                    break
        else:
            for j, t in enumerate([ t4["2"][i-n_Po212], t5_2["1"][i-n_Po212] ]):
                tmp_time = t
                if tmp_time < time_threshold:
                    if j == 1:
                        beta += 1
                    else:
                        alpha += 1
                else:
                    break
    return alpha, beta

data = []
for _ in range(10):
    tmp_alpha, tmp_beta = sim()
    data.append([ tmp_alpha, tmp_beta ])
data = np.array(data)
print("{:.0f} +/- {:.0f}".format( statistics.mean( data[:, 0] ), statistics.stdev( data[:, 0] ) ))
print("{:.0f} +/- {:.0f}".format( statistics.mean( data[:, 1] ), statistics.stdev( data[:, 1] ) ))

a = uncertainties.ufloat(statistics.mean( data[:, 0] ), statistics.stdev( data[:, 0] ))
b = uncertainties.ufloat(statistics.mean( data[:, 1] ), statistics.stdev( data[:, 1] ))
print( (b/a).nominal_value, (b/a).std_dev )