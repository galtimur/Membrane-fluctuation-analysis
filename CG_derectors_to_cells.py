#%%

'''

The programme takes as input the trajectory of the director distribution: frames containing the lipid coordinates and their director.
It discretises this distribution into cells, averaging the directors there.

The output is a file with the values of the directors in the cells

Программа принимает на вход траекторию распределения директоров: фреймы, содержащие координаты липида и их директор.
Дискретизирует это распределение по ячейкам, усредняя директора там.

На выходе - файл со значениями директоров в ячейках

'''


import numpy as np
from tqdm import tqdm
import time


def lipid_in_cell(N, L, lipid_data):
    '''Функция, которая определяет принадлежность липидов различным ячейкам (grids) мембраны'''
    # Директоры разделяются на нижний и верхний монослои
    # Заметим, что первый индекс - х, а второй - y.
    
    arr = [[[[] for i in range(N)] for i in range(N)] for i in range(2)]
    
    for lipid in lipid_data:
        [x, y, z] = lipid[1]
        xg = int(x*N/L)%N
        yg = int(y*N/L)%N
        if z > 40:
            arr[0][xg][yg].append(lipid[0][:2])
        else:
            arr[1][xg][yg].append(lipid[0][:2])
    return arr

def director_calculation_interpol(grids):   #усреднение директоров по ячейке
    
    # Заметим, что первый индекс - х, а второй - y.

    N = len(grids)    
    grid_av_norm = [[0 for i in range(N)] for i in range(N)]
    
    for lin in range(N):
        for el in range(N):
            if len(grids[lin][el]) > 0:
                grid_av_norm[lin][el] = np.average(np.array(grids[lin][el]), axis = 0)
            else:
                
                sum_dir = grids[(lin-1)%N][(el-1)%N] + grids[(lin-1)%N][el] + grids[(lin-1)%N][(el+1)%N] + grids[lin][(el-1)%N] + grids[lin][(el+1)%N] + grids[(lin+1)%N][(el-1)%N] + grids[(lin+1)%N][el] + grids[(lin+1)%N][(el+1)%N]
                if len(sum_dir) == 0:
                    return True, np.array(grid_av_norm)
                grid_av_norm[lin][el] = np.average(np.array(sum_dir), axis = 0)
                
    return False, np.array(grid_av_norm)

#%%


path = 'D:\\Timur\\MD_fluctuations\\Results\\' ## путь до файла. Поставь свой.
L = 366


N_cells = 25 # Количество ячеек, на которые мембрана разделяется
grid_av_dif_all_frames = [] #Объявляем массив для директоров бислоя

tt0 = time.time()

for i in range(1):
    
    tt = time.time()
    #filename = 'CG_frames_L=' + str(L)+'_all_part_' + str(i) + '.npy'
    filename = 'CG_frames_L=366_all_lipids_5000_frames.npy'
    file = path + filename
    part = np.load(file)
    
    for frame in part:
    
        grids = lipid_in_cell(N_cells, L, frame)
        #Проектируем директора верхнего слоя на Оху
        interpol_problem, grid_av_up = np.array(director_calculation_interpol(grids[0]))
        if interpol_problem:
            continue
        #grid_av_up_all_frames.append(grid_av_up)
        #Проектируем директора нижнего слоя на Оху
        interpol_problem, grid_av_down = np.array(director_calculation_interpol(grids[1]))
        if interpol_problem:
            continue
        #grid_av_down_all_frames.append(grid_av_down)
        #Берём половину разницы между директорами двух слоёв - получаем двумерный директор бислоя
        grid_av_dif = list(0.5*(grid_av_up - grid_av_down)) 
        grid_av_dif_all_frames.append(grid_av_dif)
        #if cc == 20: break
    print(str(i) + ' Done for ' + str(int(time.time()-tt))+' s')


print((time.time()-tt0))

grid_av_dif_all_frames = np.array(grid_av_dif_all_frames)

filename = 'CG_frames_L=' + str(L)+'_N='+str(N_cells)+'.npy'
file = path+filename

np.save(file, grid_av_dif_all_frames)
test = np.load(file)

if (np.unique((grid_av_dif_all_frames == test).flatten()))[0]:
    print('----------  OK  -------------')
