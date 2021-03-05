from datetime import time
import pyrado
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

def load_performance(timestamp):
    return np.load(f'{pyrado.TEMP_DIR}/eval/sums_{timestamp}.npy'), np.load(f'{pyrado.TEMP_DIR}/eval/names_{timestamp}.npy')

def plot_performance(sums, names, goalReward=7000, path='', showPlot=True):
    perf = np.array([ [idx, np.mean(sum), np.std(sum)] for idx, sum in enumerate(sums) ])
    horiz_line_data = np.array([goalReward for _ in perf[:,0]])
    colors = ["blue" if 'teacher' in name else "red" for name in names]
    plt.figure()
    sns.set()
    for x, y, err, color in zip(perf[:,0], perf[:,1], perf[:,2], colors):
        plt.errorbar(x, y, err, linestyle='None', marker='o', capsize=4, color=color)

    plt.xticks(perf[:,0], names, rotation=90)
    plt.plot(perf[:,0], horiz_line_data, 'r--', color="green", label="solved") 
    plt.ylabel("AVG cum reward")
    plt.legend()
    if path!='':
        plt.savefig(f'{path}plot.png')
    if showPlot:
        plt.show()
    
def load_distillation_performance(path):
    sum_files = []
    name_files = []
    for file in os.listdir(path):
        if file.endswith(".npy"):
            if file.startswith("sums_"):
                sum_files.append(file)
            elif file.startswith("names_"):
                name_files.append(file)
    
    sums = [ np.load(f'{path}/{f}') for f in sorted(sum_files) ]
    names = [ np.load(f'{path}/{f}') for f in sorted(name_files) ]
    return sums, names, sorted(name_files)

def plot_distillation_performance(env_name, timestamp, goalReward=7000, showPlot=True):
    path = f'{pyrado.TEMP_DIR}/../runs/distillation/{env_name}/{timestamp}/eval/'
    sums, names, _ = load_distillation_performance(path)
    plot_performance(sums, names, goalReward, path)

def test_files(env_name, timestamp):
    path = f'{pyrado.TEMP_DIR}/../runs/distillation/{env_name}/{timestamp}/eval/'
    _, names, name_files = load_distillation_performance(path)
    for i in range(len(names)):
        print(names[i], name_files[i])


if __name__ == "__main__":
    plot_distillation_performance('qbb','2021-03-04_09:01:04')
    plot_distillation_performance('qbb','2021-03-04_21:26:34') #goalreward?

    plot_distillation_performance('qcp-su','2021-03-03_18:50:11')
    plot_distillation_performance('qcp-su','2021-03-03_22:59:42')

    plot_distillation_performance('qq-su','2021-03-04_21:58:18', goalReward=200)
    plot_distillation_performance('qq-su','2021-03-05_01:15:55', goalReward=200)
    plot_distillation_performance('qq-su','2021-03-05_07:11:20', goalReward=200)        # richtig?
    plot_distillation_performance('qq-su','2021-03-05_09:00:51', goalReward=200)
    #plot_distillation_performance('qq-su','2021-03-05_11:25:30', goalReward=200)

