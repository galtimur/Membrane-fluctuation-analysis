'''

The programme takes frames of the directors' values in the cells. Performs a Fourier transform and averages them
Программа принимает фреймы значений директоров в ячейках. Делает их фурье преобразование и усредняет

'''


from scipy.fft import fftfreq, fft2
import numpy as np
from scipy import stats


def get_q(N, dx):
      
    # Создаём массив 2D волновых векторов, а также их модулей. На вход - число ячеек в квадратной решётке и их размер.
    q_row = np.array(2*np.pi*fftfreq(N, dx))
    
    q_vec = np.array([[[q_row[i], q_row[j]] for j in range(N)] for i in range(N)])
    q_norm = np.linalg.norm(q_vec, axis=2)
    
    '''
    !!!!!! Заметим, что здесь несколько неинтуитивно. x - первая координата, а это значит,
    !!!!!! в матрице она изменяется по вертикали, а не по горизонтали!
    !!!!!! слева направо меняются qy, сверху вниз - qx ((0,0) - левая верхняя точка)
    '''
    
	
	### Генерим тоже для вектора Q*. (именно косплексно сопряжённый тому, что у Кости написан)
    
    Q_vec = 2/dx*np.sin(q_vec*dx/2)*np.exp(1j*q_vec*dx/2) 
    Q_norm = np.linalg.norm(Q_vec, axis=2)

    return q_norm, q_vec, Q_norm, Q_vec

def get_fourier_dev(dir_array):
    ## Дисперсия амплитуд фурье
    devx_list = []
    devy_list = []
    for frame in dir_array:
        dir_q = fft2(frame, axes=(0,1), norm = 'ortho')
        devx = np.std(dir_q[:, :, 0].flatten())
        devy = np.std(dir_q[:, :, 1].flatten())
        devx_list.append(devx)
        devy_list.append(devy)
    return devx_list, devy_list

def perp(vec):
    
    ### Делаем из вектора "перепендикулярный" (без комплексного сопряжения)
    
    vec_perp = np.array([-vec[1], vec[0]])
    
    return np.conj(vec_perp)


def get_n_per_par(dir_array, q_vec, Q_vec):
    ## Каждую ячейку в каждом фрейме переводим по формуле |q|^2*|n_{||}|^2 = (n*q)^2 и |Q|^2*|n_{||}|^2 = (n*Q)^2
    ## |q|^2*|n_{_|_}|^2 = (n*q_perp)^2 и |Q|^2*|n_{_|_}|^2 = (n*Q_perp)^2
    n_qp_array = []
    n_Qp_array = []
    Hq_array = []
    
    n_qperp_array = []
    n_Qperp_array = []
    
    N = np.shape(dir_array)[1]

    for frame in dir_array:
        def_q = fft2(frame, axes=(0,1))
        
        dir_q = def_q[:,:,[0,1]]
        Hq = def_q[:,:,2]
       
        norm = np.array([[(np.abs(np.dot(q_vec[i,j], dir_q[i,j])))**2 for j in range(N)] for i in range(N)])
        n_qp_array.append(norm)
        
        norm = np.array([[(np.abs(np.dot(Q_vec[i,j], dir_q[i,j])))**2 for j in range(N)] for i in range(N)])
        n_Qp_array.append(norm)
        
        norm = np.array([[(np.abs(np.dot(perp(q_vec[i,j]), dir_q[i,j])))**2 for j in range(N)] for i in range(N)])
        n_qperp_array.append(norm)
        
        norm = np.array([[(np.abs(np.dot(perp(Q_vec[i,j]), dir_q[i,j])))**2 for j in range(N)] for i in range(N)])
        n_Qperp_array.append(norm)
        
        norm = np.array([[(np.abs(Hq[i,j]))**2 for j in range(N)] for i in range(N)])
        Hq_array.append(norm)
                 
        
    return np.array(n_qp_array), np.array(n_Qp_array), np.array(n_qperp_array), np.array(n_Qperp_array), np.array(Hq_array)


def average_frames(np_array, q_norm, L):
    ## усредняем по всем фреймам и разворачиваем в линейный массив
    np_av = np.average(np_array , axis=0)
    N = np.shape(np_av)[0]
    ## разворачиваем массив
    ### Убираем первый элемент, который [0, 0]
    ### Домножаем на нормировку
    npSq_list = (np_av.reshape(N**2)[1:])*(L**2)/(N**4)
    q_norm_list = q_norm.reshape(N**2)[1:]
    
    ### На выходе список проекций, соответсвующий набору норм векторов и матрица проекций, усреднённая по фреймам.
    
    return npSq_list, q_norm_list, np_av


def get_np_q(MC_frames, L):
    
    ## Считываем данные. Считаем фурье и проекции. Усредняем по фреймам
    ## Финальная структура - усреднённые проекции, в каждой из которых NxN ячеек, в каждой из которых двумерный вектор.
    ## !!! npq_av, npQ_av не передаём, .т.к они не используются. Это не усреднённые по q данные. Они не имеют смысла, кроме каких-то тестов.
    
    N = np.shape(MC_frames)[1]
    dx = L/N
    q_norm, q_vec, Q_norm, Q_vec = get_q(N, dx)
    
    npq, npQ, nperp_q, nperp_Q, Hq = get_n_per_par(MC_frames, q_vec, Q_vec)
    
    npq_list, qnorm_list, npq_av = average_frames(npq, q_norm, L)
    npQ_list, Qnorm_list, npQ_av = average_frames(npQ, Q_norm, L)

    nperp_q_list, qnorm_list, nperp_q_av = average_frames(nperp_q, q_norm, L)
    nperp_Q_list, Qnorm_list, nperp_Q_av = average_frames(nperp_Q, Q_norm, L)
    
    Hq_list, Qnorm_list, Hq_av = average_frames(Hq, Q_norm, L)
    
    return npq_list, qnorm_list, npQ_list, Qnorm_list, nperp_q_list, nperp_Q_list, Hq_list

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

def analyze_2D_fluct(MC_frames, L):
    
    ### сборка анализа
    
    npq_list, qnorm_list, npQ_list, Qnorm_list, nperp_q_list, nperp_Q_list, Hq_list = get_np_q(MC_frames, L)
    
    ### усреднение в рамках одинаковых q (Q)
    qnorm, npq = average_q(qnorm_list, npq_list)
    Qnorm, npQ = average_q(Qnorm_list, npQ_list)
    
    qnorm, nperp_q = average_q(qnorm_list, nperp_q_list)
    Qnorm, nperp_Q = average_q(Qnorm_list, nperp_Q_list)
    
    qnorm, Hq = average_q(qnorm_list, Hq_list)
    Qnorm, HQ = average_q(Qnorm_list, Hq_list)
       
    ### Убираем q^2, появившийся при вычислении проекций на q
    
    npq = npq/qnorm**(2)
    npQ = npQ/Qnorm**(2)
    nperp_q = nperp_q/qnorm**(2)
    nperp_Q = nperp_Q/Qnorm**(2)
    
    return qnorm, npq, Qnorm, npQ, nperp_q, nperp_Q, Hq, HQ


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



#%%