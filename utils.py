import numpy as np 
import matplotlib.pyplot as plt 

def plot_SR_synthetic_data(SR_stat, 
                           window_width,
                           threshold,
                           add_obs = 25,
                           model_used = 'linear model',
                           ranks_used = 'no ranks'):
    cp_lin = 0
    for t, i in enumerate(SR_stat):
        if i > threshold:
            cp_lin = t
            break

    plt.figure(figsize = (9, 3))
    plt.plot(np.arange(475,(500+add_obs)), SR_stat[425:(500-window_width+add_obs)], color = 'blue', label = 'SR statistic')
    plt.grid(alpha = 0.5)
    plt.axvline(cp_lin+window_width, color = 'green', linestyle = 'dotted', label = r'$\tau$')
    plt.axvline(500, color = 'red', linestyle = 'dashdot', label = r'$\theta$')
    plt.axhline(threshold, label = 'A', linestyle = '--', color = 'orange')
    plt.ylabel('SR statistic')
    plt.xlabel('time')
    plt.title(f'{model_used}, {ranks_used}')
    plt.xticks(np.arange(475,(500+add_obs+1), 4))
    plt.xlim(475, (500+add_obs))
    plt.legend(loc = (.01, .55))
    plt.show()