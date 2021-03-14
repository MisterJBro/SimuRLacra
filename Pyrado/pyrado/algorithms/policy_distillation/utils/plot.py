from datetime import time
import pyrado
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from pyrado.algorithms.policy_distillation.utils.load import ask_for_packlist, load_packed_teachers

def load_performance(timestamp):
    return np.load(f'{pyrado.TEMP_DIR}/eval/sums_{timestamp}.npy'), np.load(f'{pyrado.TEMP_DIR}/eval/names_{timestamp}.npy')

def cheat(goalReward=1500*0.7, path='', showPlot=True):
    perf =  np.array([[ 0,  7.1066e+02,  2.9794e+02]
                    , [ 1,  1056.7812230607724,  155.69650159391375]        #Endsumme (student_on_teacher_env_0 from 100 reps ): MEAN = 1056.7812230607724 STD = 155.69650159391375 MIN = 762.36048928932 MAX = 1477.9507441726887 MEDIAN = 1043.0720679559322
                    , [ 2,  404.81211000983353,  150.3645893504437]         #Endsumme (student_on_teacher_env_1 from 100 reps ): MEAN = 404.81211000983353 STD = 150.3645893504437 MIN = 247.2540605991172 MAX = 1211.036494950129 MEDIAN = 374.3344905912277
                    , [ 3,  1156.9195305665196,  221.57916423864864]        #Endsumme (student_on_teacher_env_2 from 100 reps ): MEAN = 1156.9195305665196 STD = 221.57916423864864 MIN = 359.7573451829713 MAX = 1476.7499779071557 MEDIAN = 1159.8194143110559
                    , [ 4,  1.1661e+03,  1.8607e+02]
                    , [ 5,  9.0869e+02,  3.1151e+02]
                    , [ 6,  8.9505e+02,  3.2975e+02]
                    , [ 7,  5.4817e+02,  2.5356e+02]
                    , [ 8,  7.3062e+02,  8.9309e+01]
                    , [ 9,  5.8388e+02,  2.9580e+02]
                    , [ 10,  3.8153e+02,  2.7137e+02]
                    , [ 11,  6.4126e+02,  1.5832e+02]
                    , [ 12,  1.0840e+03,  1.5675e+02]
                    , [ 13,  7.3985e+02,  1.0328e+02]
                    , [ 14,  4.3419e+02,  1.6420e+02]
                    , [ 15,  4.6804e+02,  1.4072e+02]
                    , [ 16,  6.5009e+02,  1.5431e+02]])
    names = np.array([  ['student_after'],
                        ['student_on_teacher_env_0'], 
                        ['student_on_teacher_env_1'], 
                        ['student_on_teacher_env_2'], 
                        ['student_on_teacher_env_3'], 
                        ['student_on_teacher_env_4'], 
                        ['student_on_teacher_env_5'], 
                        ['student_on_teacher_env_6'], 
                        ['student_on_teacher_env_7'], 
                        ['student_on_random_env_0'],
                        ['student_on_random_env_1'],
                        ['student_on_random_env_2'],
                        ['student_on_random_env_3'],
                        ['student_on_random_env_4'],
                        ['student_on_random_env_5'],
                        ['student_on_random_env_6'],
                        ['student_on_random_env_7'],])
    names = np.array([  'student',
                        'teacher 0', 
                        'teacher 1', 
                        'teacher 2', 
                        'teacher 3', 
                        'teacher 4', 
                        'teacher 5', 
                        'teacher 6', 
                        'teacher 7', 
                        'random 0',
                        'random 1',
                        'random 2',
                        'random 3',
                        'random 4',
                        'random 5',
                        'random 6',
                        'random 7'])

    horiz_line_data = np.array([goalReward for _ in perf[:,0]])
    colors = ["blue" if 'student' in name else "red" for name in names]
    plt.figure()
    #sns.set()
    for x, y, err, color in zip(perf[:,0], perf[:,1], perf[:,2], colors):
        plt.errorbar(x, y, err, linestyle='None', marker='o', capsize=4, color=color)

    plt.xticks(perf[:,0], names, rotation=90)
    plt.plot(perf[:,0], horiz_line_data, 'r--', color="green", label="solved") 
    plt.ylabel("AVG cum reward")
    plt.xlabel("environment")
    plt.margins(0.03)
    plt.legend()
    if path!='':
        plt.savefig(f'{path}plot.png')
        plt.savefig(f'{path}plot.pdf')
    if showPlot:
        plt.show()


def plot_performance(sums, names, goalReward=7000, path='', showPlot=True):
    perf = np.array([ [idx, np.mean(sum), np.std(sum)] for idx, sum in enumerate(sums) ])
    
    print(perf)
    print(np.array(names).reshape(-1))
    names = np.array(names).reshape(-1)

    horiz_line_data = np.array([goalReward for _ in perf[:,0]])
    colors = ["blue" if 'teacher' in name else "red" for name in names]
    plt.figure()
    #sns.set()
    for x, y, err, color in zip(perf[:,0], perf[:,1], perf[:,2], colors):
        plt.errorbar(x, y, err, linestyle='None', marker='o', capsize=4, color=color)

    plt.xticks(perf[:,0], names, rotation=90)
    plt.plot(perf[:,0], horiz_line_data, 'r--', color="green", label="solved") 
    plt.ylabel("AVG cum reward")
    plt.xlabel("policy/environment")
    plt.margins(0.03)
    plt.legend()
    if path!='':
        plt.savefig(f'{path}plot.png')
        plt.savefig(f'{path}plot.pdf')
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
    #cheat(goalReward, path)

def plot_pack_performance(env_name, teacher_count, goalReward=1500*0.7):
    packlist = [[32],["1500_250_0"]]#ask_for_packlist()
    _, _, _, _, _, _, sums, names = load_packed_teachers(env_name, packlist, teacher_count)

    for s in sums:
        print(np.shape(s))

    plot_performance(np.array(sums), np.array(names), goalReward)   #path

def test_files(env_name, timestamp):
    path = f'{pyrado.TEMP_DIR}/../runs/distillation/{env_name}/{timestamp}/eval/'
    _, names, name_files = load_distillation_performance(path)
    for i in range(len(names)):
        print(names[i], name_files[i])


if __name__ == "__main__":
    """plot_distillation_performance('qbb','2021-03-04_09:01:04')
    plot_distillation_performance('qbb','2021-03-04_21:26:34') #goalreward?

    plot_distillation_performance('qcp-su','2021-03-03_18:50:11')
    plot_distillation_performance('qcp-su','2021-03-03_22:59:42')

    plot_distillation_performance('qq-su','2021-03-04_21:58:18', goalReward=200)
    plot_distillation_performance('qq-su','2021-03-05_01:15:55', goalReward=200)
    plot_distillation_performance('qq-su','2021-03-05_07:11:20', goalReward=200)        # richtig?
    plot_distillation_performance('qq-su','2021-03-05_09:00:51', goalReward=200)
    #plot_distillation_performance('qq-su','2021-03-05_11:25:30', goalReward=200)
    """
    # draft
    #plot_distillation_performance('qcp-su','2021-03-10_18:12:22', goalReward=1500*0.7)
    #plot_distillation_performance('qcp-su', 'teacher', goalReward=1500*0.7) 

    #
    plot_pack_performance('qq-su', 32)  #32 1500_250_0
    #plot_pack_performance('qcp-su', 32)  #32 1500_250_0
