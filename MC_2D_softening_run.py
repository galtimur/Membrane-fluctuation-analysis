import numpy as np
from random import uniform, randint, random, seed
import time


# def en_fun(curv):
#     if abs(curv) < 1.5708/2:
#         en = (1/2*np.sin(2*curv))**2
#     else:
#         #en = curv**2
#         en = 1/4 + (abs(curv)-1.5708/2)/2
#     return en

# def en_fun(curv):
#     en = (0.25*np.log(1 + abs(curv)/0.25))**2
#     return en

def en_fun(curv):
    if abs(curv) < 0.6:
        en = curv**2
    else:
        en = 0.36+1.2*(abs(curv)-0.6)
    return en

def run_MC_2D(N_frames, period, delta, N, L, B, Kt, Ktw, soft, warm = 2, do_another = False):

	'''
	run the Monte Carlo simulations
	'''

    '''MC-симуляция решётки с "изгибом" и тилтом. В сумме по изгибу дивергенция определяется с плюсом: (n[i] - n[i+1]) '''
    seed(1)
    #Управляемые параметры симуляции:

    N_steps = (N_frames + warm)*period
    dx = L/N
    
    # soft = False
    # N_steps = 10
    # dx = 16/11
    # N = 11
    # delta = 0.05
    # B = 10

    
    #Задаём стартовую позицию двумерного массива двумерных векторов:
    n = np.zeros((N,N,2))
    H = np.zeros((N,N))
    ### Расширяем массив, чтобы вместо скаляра в каждой ячейке был вектор длиной 1
    H = np.expand_dims(H.copy(), axis=2)
    
    tt = time.time()
    
    #Основной блок программы - симуляция тепловых флуктуаций (thermal fluctuations):
    MC_frames = [] #Объявляем лист кадров симуляции тепловых флуктуаций
    En_old = 0
    #en_lst = []
    #curv_lst = []
    '''
    !!!!!! Заметим, что здесь несколько неинтуитивно. x - первая координата, а это значит,
    !!!!!! в матрице она изменяется по вертикали, а не по горизонтали!
    '''


    
    for t in range(N_steps):
        
        En_old0 = En_old
        i_n = randint(0, N-1)
        j_n = randint(0, N-1)
        # ## n_x projections
   
        dnx = delta*uniform(-1., 1.)
        dny = delta*uniform(-1., 1.)


        n_new_x, n_new_y = n[i_n, j_n, 0] + dnx, n[i_n, j_n, 1] + dny
        n[i_n, j_n] = [n_new_x, n_new_y]
        
        
        en = 0
        for i in range(N):
            for j in range(N):
                curv = ((n[(i+1)%N, j, 0] - n[i, j, 0])/dx + 0*(n[i, (j+1)%N, 1] - n[i, j, 1])/dx)
                if soft:                
                    en = en + en_fun(curv) + 1.2*(n[i, j, 0]**2 + n[i, j, 1]**2)
                else:
                    en = en + curv**2 + 1.2*(n[i, j, 0]**2 + n[i, j, 1]**2)
        
        #curv_lst.append(curv)

        en = B/2*dx*dx*en
        
        dE = en - En_old
        En_old = en
        
        
        ### Сдвигаем директор
        
        if (dE > 0):
            p = np.exp(-dE)
            if (random() > p):
                n_new_x, n_new_y = n[i_n, j_n, 0] - dnx, n[i_n, j_n, 1] - dny
                n[i_n, j_n] = [n_new_x, n_new_y]
                En_old = En_old0
                #curv_lst.pop(-1)
                #en = en + dE
                       
        if (t % period == 0):
            ### Каждый период записываем деформации мембраны
            ### Расширяем массив, чтобы вместо скаляра в каждой ячейке был вектор длиной 1
            ### конкатенируем, так, что первые два элемента вектора в ячейке - это директор, третий - высота.
            deform = np.concatenate((n.copy(), H), axis=2)
            MC_frames.append(deform)
            #en_lst.append(en)
    
    MC_frames = np.array(MC_frames[warm:])
    #en_lst = en_lst[warm:]
        
    print('Simulation time = {:0.1f} s.'.format(time.time()-tt))
    
    
    return MC_frames#, curv_lst#, np.array(en_lst)

    '''
    
    ### центрируем массив, вычитая среднее на каждом фрейме.
    MC_frames = []
    for frame in MC_frames0:
        mean_np = np.mean(frame, axis = (0,1))
        snap_new = frame - mean_np
        MC_frames.append(snap_new)
    
    MC_frames = np.array(MC_frames)
    
    '''
    
    '''
    ### Проверка, что все средние равны нулю.
    means = []
    for frame in MC_frames0:
        mean_np = np.mean(frame, axis = (0,1))
        means.append(mean_np)
    MC_frames = MC_frames0
    '''

    '''
      
    ### Проверка на то, что энергия правильно считалась
    
    import matplotlib.pyplot as plt
    
    en_lst2 = []
    for frame in MC_frames:
        en = 0
        for i in range(N):
            for j in range(N):
                en = en + B/2*dx*dx*((frame[i, j, 0] - frame[i-1, j, 0])/dx + (frame[i, j, 1] - frame[i, j-1, 1])/dx)**2
                en = en + Kt/2*dx*dx*((frame[i, j, 0])**2 + (frame[i, j, 1])**2)
                en = en + Ktw/2*dx*dx*((frame[i, j, 1] - frame[i-1, j, 1])/dx - (frame[i, j, 0] - frame[i, j-1, 0])/dx)**2
        en_lst2.append(en)
    
    plt.figure(0)
    plt.plot(np.array(en_lst)+0.2, 'o')
    plt.plot(en_lst2, 'o')
    plt.plot([N**2]*N_frames, 'r')
    plt.show()
    
    '''
    
    '''   
    
    ### Считаем и строим дисперсию.
    devx_list = []
    devy_list = []
    for frame in MC_frames:
        devx = np.std(frame[:, :, 0].flatten())
        devy = np.std(frame[:, :, 1].flatten())
        devx_list.append(devx)
        devy_list.append(devy)
    
    plt.figure(1)
    plt.plot(devx_list, 'o')
    plt.plot(devy_list, 'o')
    plt.show()
    
    #plt.plot(MC_frames[1900, 6,:, 0])
    #plt.plot(MC_frames[1900, 6,:, 1])
    #plt.show()
    
    '''

''' При необходимости можно включить более явный способ расчёта энергии '''

    # en_lst = []
    # en2_lst = []
    # en = 0
    # en2 = 0
    # dE2 = 0
    
        # if do_another:
 
        #     i_range = [i_n, (i_n-1)%N]
        #     j_range = [j_n, (j_n-1)%N]
                  
        #     en_old = 0        
        #     for i in i_range:
        #         for j in j_range:
        #             en_old = en_old + B/2*dx*dx*((-n_p[i, j, 0] + n_p[(i+1)%N, j, 0])/dx + (-n_p[i, j, 1] + n_p[i, (j+1)%N, 1])/dx)**2
        #             en_old = en_old + Kt/2*dx*dx*((n_p[i, j, 0])**2 + (n_p[i, j, 1])**2)
            
        #     [n_old_x, n_old_y] = n_p[i_n, j_n]
        #     n_p[i_n, j_n] = [n_p[i_n, j_n, 0] + dnx, n_p[i_n, j_n, 1] + dny]
        #     en_new = 0
        #     for i in i_range:
        #         for j in j_range:
        #             en_new = en_new + B/2*dx*dx*((-n_p[i, j, 0] + n_p[(i+1)%N, j, 0])/dx + (-n_p[i, j, 1] + n_p[i, (j+1)%N, 1])/dx)**2
        #             en_new = en_new + Kt/2*dx*dx*((n_p[i, j, 0])**2 + (n_p[i, j, 1])**2)
            
        #     n_p[i_n, j_n] = [n_old_x, n_old_y]
            
        #     dE2 = en_new - en_old
        
        