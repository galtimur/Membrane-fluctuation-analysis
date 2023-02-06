'''

The programme takes frames of the directors' values in the cells. Performs a Fourier transform and averages them

Программа принимает фреймы значений директоров в ячейках. Делает их фурье преобразование и усредняет

'''

#from scipy.fft import fftfreq, fft2
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
import time



def ft_cont(fx, coords, q_lst, L, dx):
    
    '''
    1D Фурье преобразование функции fx на непрерывном пространстве
    '''
    qx_mat = -1j*np.outer(coords, q_lst)
    qx_mat_exp = np.exp(qx_mat)    
    fq = dx/(2*np.pi)*np.dot(fx, qx_mat_exp)
    
    return fq



def ft2_cont(fxy, q_lst):
    
    '''
    2D Фурье преобразование функции fx на непрерывном квадратном пространстве
    На вход подаём функцию в виде списка [x, y, f[x, y]]
    '''
    
    N_points = fxy.shape[0]
    f = fxy[:,2]
    x = fxy[:,0]
    y = fxy[:,1]
    qx_mat = -1j*np.outer(x, q_lst)
    qy_mat = -1j*np.outer(y, q_lst)
    fq = np.einsum('k, ki, kj -> ij', f, np.exp(qx_mat), np.exp(qy_mat))
    
    return fq/N_points


def get_q(N, L, dx):
      
    # Создаём массив 2D волновых векторов, а также их модулей. На вход - число ячеек в квадратной решётке и их размер.
    ## Минимальный q = 1/L, максимальный - 1/dx. Заметим, что N и dx развязались. dx - размер молекулы, N - число волновых векторов 
    q_row = 2*np.pi*np.linspace(-0.2*L/dx, 0.2*L/dx, N)/L
    
    q_vec = np.array([[[q_row[i], q_row[j]] for j in range(N)] for i in range(N)])
    q_norm = np.linalg.norm(q_vec, axis=2)
    
    '''
    !!!!!! Заметим, что здесь несколько неинтуитивно. x - первая координата, а это значит,
    !!!!!! в матрице она изменяется по вертикали, а не по горизонтали!
    !!!!!! слева направо меняются qy, сверху вниз - qx ((0,0) - левая верхняя точка)
    '''
    
    return q_norm, q_vec, q_row


def perp(vec):
    
    ### Делаем из вектора "перепендикулярный" (без комплексного сопряжения)
    
    vec_perp = np.array([-vec[1], vec[0]])
    
    return np.conj(vec_perp)


def make_fourier(frames, q_vec, q_row, N):
    ## Каждую ячейку в каждом фрейме переводим по формуле |q|^2*|n_{||}|^2 = (n*q)^2
    ## |q|^2*|n_{_|_}|^2 = (n*q_perp)^2
    
    ### структура frames в непрерывном случае
    
    global dir_q, dirq_x, dirq_y
    
    n_qp_array = []
    Hq_array = []
    
    n_qperp_array = []

    for frame in frames:
        
        dir_x = frame[:,0,0]
        dir_y = frame[:,0,1]
        x = frame[:,1,0]
        y = frame[:,1,1]
        z = frame[:,1,2]
        
        dir_x = np.transpose([x, y, dir_x])
        dir_y = np.transpose([x, y, dir_y])
        H = np.transpose([x, y, z])
        
        dirq_x = ft2_cont(dir_x, q_row)
        dirq_y = ft2_cont(dir_y, q_row)
        Hq = ft2_cont(H, q_row)
        
        dir_q = np.dstack((dirq_x, dirq_y))
        
        norm = np.array([[(np.abs(np.dot(q_vec[i,j], dir_q[i,j])))**2 for j in range(N)] for i in range(N)])
        n_qp_array.append(norm)
        
        norm = np.array([[(np.abs(np.dot(perp(q_vec[i,j]), dir_q[i,j])))**2 for j in range(N)] for i in range(N)])
        n_qperp_array.append(norm)
        
        norm = np.array([[(np.abs(Hq[i,j]))**2 for j in range(N)] for i in range(N)])
        Hq_array.append(norm)
                 
        
    return np.array(n_qp_array), np.array(n_qperp_array), np.array(Hq_array)


def average_frames(np_array, q_norm, L):
    ## усредняем по всем фреймам и разворачиваем в линейный массив
    np_av = np.average(np_array , axis=0)
    N = np.shape(np_av)[0]
    ## разворачиваем массив
    ### Убираем первый элемент, который [0, 0]
    ### Домножаем на нормировку
    npSq_list = (np_av.reshape(N**2)[1:])*(L**2)#/(N**4)
    q_norm_list = q_norm.reshape(N**2)[1:]
    
    ### На выходе список проекций, соответсвующий набору норм векторов и матрица проекций, усреднённая по фреймам.
    
    return npSq_list, q_norm_list, np_av


def get_np_q(frames, L, N, dx):
    
    ## Считываем данные. Считаем фурье и проекции. Усредняем по фреймам
    ## Финальная структура - усреднённые проекции, в каждой из которых NxN ячеек, в каждой из которых двумерный вектор.
    ## !!! npq_av, npQ_av не передаём, .т.к они не используются. Это не усреднённые по q данные. Они не имеют смысла, кроме каких-то тестов.
    
    q_norm, q_vec, q_row = get_q(N, L, dx)
    
    npq, nperp_q, Hq = make_fourier(frames, q_vec, q_row, N)  
    npq_list, qnorm_list, npq_av = average_frames(npq, q_norm, L)
    nperp_q_list, qnorm_list, nperp_q_av = average_frames(nperp_q, q_norm, L)
    Hq_list, qnorm_list, Hq_av = average_frames(Hq, q_norm, L)
    
    return npq_list, qnorm_list, nperp_q_list, Hq_list

def average_q(q_list, n_list):
    
    ### берём значения nq и усредняем их при одинаковых q
    
    points = np.array([q_list, n_list]).T
    ## Сортируем по первому элементу (q)
    points_sorted = (points[points[:,0].argsort()]).copy()
    n_split = np.split(points_sorted[:,1], np.unique(points_sorted[:, 0], return_index = True)[1][1:])   
    n_mean = list(map(np.mean, n_split))
    
    return np.array(np.unique(q_list)), np.array(n_mean)

def bin_average(q_list, n_list, dq):
    
    bin_num = int((q_list[-1] - q_list[0])//dq)
    bin_means, bin_edges, binnumber = stats.binned_statistic(q_list, n_list, statistic='mean', bins=bin_num)
    bin_width = (bin_edges[1] - bin_edges[0])
    bin_centers = bin_edges[1:] - bin_width/2

    return bin_centers, bin_means

def analyze_2D_fluct(frames, L, N, dx):
    
    
    ### сборка анализа
    
    npq_list, qnorm_list, nperp_q_list, Hq_list = get_np_q(frames, L, N, dx)
    
    ### усреднение в рамках одинаковых q (Q)
    qnorm, npq = average_q(qnorm_list, npq_list)
    qnorm, nperp_q = average_q(qnorm_list, nperp_q_list)
    qnorm, Hq = average_q(qnorm_list, Hq_list)
       
    ### Убираем q^2, появившийся при вычислении проекций на q
    
    npq = npq/qnorm**(2)
    nperp_q = nperp_q/qnorm**(2)
    
    return qnorm, npq, nperp_q, Hq


#### Итого:
#### 1. Делаем фурье
#### 2. Усредняем по фреймам
#### 3. Усредняем по проекциям с одинаковым q


#%%


if __name__ == "__main__":


    from scipy.fft import fft
    import matplotlib.pyplot as plt    

    #def testfunction():    

    n = 50
    x_lst = np.linspace(0, 2*np.pi, num=n+1)[:-1]
    nums = np.array(range(n))/n
    q_lst = fftfreq(n)
    dir_test = np.sin(x_lst)
    dir_q = fft(dir_test)
    
    dir_q_an = np.array([np.dot(dir_test, np.exp(-2*np.pi*1j*i*nums)) for i in nums])
    np.array([np.dot(dir_test, np.exp(-2*np.pi*1j*i*q_lst)) for i in nums])
    
    np.concatenate((nums[:n//2], -np.flip(nums[1:n//2+1])))
    nums[:n//2]
    
    dir_q_abs = abs(dir_q)
    
    plt.figure(0)
    plt.plot(x_lst, dir_test)
    plt.figure(1)
    plt.plot(q_lst, dir_q_abs, 'o')
    
    print(2*np.pi - x_lst[-1])
    
    #return

#testfunction()


#%%

'''


# Тест 1D фурье

coords = np.linspace(-np.pi, np.pi, 201)
fx_test = np.sin(coords)
dx = coords[2]-coords[1]
L = 2*np.pi
q_lst = 2*np.pi*np.linspace(-10, 10, 21)/L

plt.plot(coords, fx_test)
plt.show()

tt = time.time()
qq = ft_cont(fx_test, coords, q_lst, L, dx=dx)
print(time.time() - tt)

plt.plot(q_lst, np.real(qq), marker='o')
plt.plot(q_lst, np.imag(qq), marker='o')
plt.show()

'''
'''

# Тест 2D фурье ## по N сходимость слабая, но если составить ансамбль, то лучше становится

N = 50
L = 5.1
dx = L/N
fq_im = np.zeros((40, 21, 21))

for i in range(40):

    #coords = np.linspace(-np.pi, np.pi, N)
    coords = L*(np.random.rand(N)-0.5)
    fxy = np.array([[[x, y, np.sin(x*2*np.pi/L)*np.cos(y*2*np.pi/L) + np.sin(2*x*2*np.pi/L) + np.sin(3*y*2*np.pi/L)] for y in coords] for x in coords])
    #fxy = np.array([[[x, y, np.sin(3*y)] for y in coords] for x in coords])
    fxy = fxy.reshape(N*N, 3)
    q_lst = 2*np.pi*np.linspace(-10, 10, 21)/L
    
    
#    tt = time.time()
    fq = ft2_cont(fxy, q_lst)
    fq_im[i] = np.imag(fq)
#    print(time.time() - tt)
    #print(fq_im[10,7])


fq_im_av = np.mean(fq_im, axis=0)
print('average = ', fq_im_av[10,7])
print('std = ', np.std(fq_im[:,10,7]))

'''
#%%

# L0 = 16
# N0 = 10
# dx0 = 0.8



# file_name = "D:\Timur\MD_Fluctuations\Results\CG_frames_L=366_all_lipids_5000_frames.npy"
# frames = np.load(file_name)
# frames_test = frames[:1000]

# qnorm, npq, nperp_q, Hq = analyze_2D_fluct(frames_test, L = L0, N = N0, dx = dx0)

# #%%

# plt.plot(qnorm, npq, 'o')
# plt.show()

# #%%
