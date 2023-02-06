'''

The program takes as input a file with frames of directors' values in cells
The analyze_2D_fluct function performs a Fourier transform and averages them
Then results are plotted.
If the do_MC flag is set, it starts a Monte Carlo simulation.

Программа принимает на вход файл с фреймами значений директоров в ячейках
С помощью функции analyze_2D_fluct делает их фурье преобразование и усредняет
Потом строятся графики результатов.
Если стоит флаг do_MC, то запускает монте-карло симуляцию.

'''


import numpy as np
import matplotlib.pyplot as plt
from MC_2D_softening_run import run_MC_2D
from fluctuation_analysis_2D import analyze_2D_fluct


L0 = 16
B0 = 10
Kt0 = 8
Ktw0 = 5
beta_twist = Ktw0/Kt0
beta_tilt = B0/Kt0
N0 = 11
N_periods = 1000

plot_MC = False
do_MC = False
plot_CG = True

file_name_save = 'Results/MC_frames_0.1k_Kt=8_B=10_Ktw=5_L=16_test'
file_name_save = 'Results/MC_frames_5k_direct_soft_log_en_B=10_N=11.npy'
file_name_load = 'Results/MC_frames_10k_Kt=10_B=10_Ktw=0_L=16_N=11.npy'
file_name_load = 'Results/MC_frames_5k_direct_soft_en_B=10_N=11.npy'
file_name_load = 'Results/MC_frames_1k_direct_soft_abs_en_B=10_N=11.npy'


if do_MC:
    MC_frames = run_MC_2D(N_frames = N_periods, period = 25*N0*N0, delta = 0.05, warm = 50, soft = True, N = N0, L = L0, B = B0, Kt = Kt0, Ktw = Ktw0)
    np.save(file_name_save, MC_frames)
elif plot_MC:
    MC_frames = np.load(file_name_load)

if plot_MC:
    
#    qq1[100,1,2,1]
#    MC_frames[100,1,2,1]
    
#    qq1 = MC_frames
    
    # Меняем местами nx и ny, чтобы проверить
    # MC_frames_n = MC_frames[:,:,:,:2]
    # MC_frames_n = np.flip(MC_frames_n, axis=3)
    # data_shape = MC_frames_n.shape
    # data_shape_add = list(data_shape)[:-1] + [1]
    # MC_frames = np.append(MC_frames_n, np.zeros(data_shape_add), axis=-1)

    qnorm, npq, Qnorm, npQ, nperp_q, nperp_Q, Hq, HQ = analyze_2D_fluct(MC_frames, L = L0)
    
   
    plt.figure(1)
    plt.title("1/np q for Mone-Carlo \n Loaded from \n" + file_name_load)
    plt.plot(qnorm, 1/((qnorm**2)*npq), 'o')
    plt.axhline(y = B0, color='r', linestyle='-')
    #plt.ylim((0,1.5*B0))
    plt.show()
    
    #plt.figure(2)
    plt.title("1/np Q for Mone-Carlo \n Loaded from \n" + file_name_load)
    plt.plot(Qnorm, 1/((Qnorm**2)*npQ), 'o')
    plt.axhline(y = B0, color='r', linestyle='-')
    plt.ylim((0,1.5*B0))
    plt.show()
    
    
    #plt.figure(2)
    plt.title("1/np Q for Mone-Carlo \n Loaded from \n" + file_name_load)
    plt.plot(qnorm, (npq), 'o')
    plt.show()
       
    # plt.figure(3)
    # plt.title("1/n_perp q for Mone-Carlo \n Loaded from \n" + file_name_load)
    # plt.plot(qnorm, 1/nperp_q, 'o')
    # plt.axhline(y = Kt0, color='r', linestyle='-')
    # plt.ylim((0,1.5*B0))
    # plt.show()
    
    #plt.figure(4)
    plt.title("1/n_perp Q for Mone-Carlo \n Loaded from \n" + file_name_load)
    plt.plot(Qnorm, 1/(nperp_Q*(1 + beta_twist*Qnorm**(2))), 'o')
    plt.axhline(y = Kt0, color='r', linestyle='-')
    plt.ylim((0,1.5*Kt0))
    plt.show()
    
    #plt.figure(5)
    plt.title("1/Hq for Mone-Carlo \n Loaded from \n" + file_name_load)
    plt.plot(Qnorm, (1 + (Qnorm**2)*B0/Kt0)/((Qnorm**4)*Hq), 'o')
    plt.axhline(y = B0, color='r', linestyle='-')
    plt.ylim((0,1.5*B0))
    plt.show()

if plot_CG:

#qq_av = CG_frames_av    


    L0 = 36.6
    #Kt0 = 9
    path = 'D:\\Timur\\MD_fluctuations\\Results\\'
    file_name_load = path + 'CG_frames_L=366_N=25.npy'
    CG_frames = np.load(file_name_load)
    CG_frames_copy=CG_frames[80000::10]
    CG_frames = CG_frames[80000:]

    n_av = 10
    num_split = len(CG_frames)//n_av
    CG_frames_split = np.array(np.array_split(CG_frames[:n_av*num_split], num_split))
    CG_frames_av = np.mean(CG_frames_split, axis=1)


    data_shape = CG_frames_av.shape
    data_shape_add = list(data_shape)[:-1] + [1]
    #CG_frames = CG_frames.reshape(2001, 11, 11, 2)
    ### Добиваю нулями, чтобы размерность совпадала с монте-карло (там ещё высота третьим элементом)
    CG_frames_fin = np.append(CG_frames_av, np.zeros(data_shape_add), axis=-1)
    
    
    data_shape = CG_frames_copy.shape
    data_shape_add = list(data_shape)[:-1] + [1]
    #CG_frames = CG_frames.reshape(2001, 11, 11, 2)
    ### Добиваю нулями, чтобы размерность совпадала с монте-карло (там ещё высота третьим элементом)
    CG_frames_copy_fin = np.append(CG_frames_copy, np.zeros(data_shape_add), axis=-1)
    

    qnorm, npq, Qnorm, npQ, nperp_q, nperp_Q, Hq, HQ  = analyze_2D_fluct(CG_frames_copy_fin, L = L0)
  
    # plt.figure(1)
    # plt.title("1/(npq*q^2) for Coarse Grain \n Loaded from \n" + file_name_load)
    # plt.plot(qnorm, 1/((qnorm**2)*npq), 'o')
    # plt.axhline(y = 2*B0, color='r', linestyle='-')
    # plt.ylim((0,1.5*2*B0))
    # plt.show()

    
    plt.title("1/(npQ*Q^2) for Coarse Grain \n Loaded from \n" + file_name_load)
    plt.plot(Qnorm, 1/((Qnorm**2)*npQ), 'o')
    plt.axhline(y = 2*B0, color='r', linestyle='-')
    plt.ylim((0,2*2*B0))
    plt.show()
    
    plt.title("(npQ*Q^2) for Coarse Grain \n Loaded from \n" + file_name_load)
    plt.plot(Qnorm, (Qnorm**2)*(npQ), 'o')
    plt.show()    

    plt.title("(npQ) for Coarse Grain \n Loaded from \n" + file_name_load)
    plt.plot(qnorm, (npq), 'o')
    plt.ylim((0,0.07))
    plt.show()    



npqinv18 = (qnorm, 1/((qnorm**2)*npq))
npQinv18 = (Qnorm, 1/((Qnorm**2)*npQ))
npQ18 = (Qnorm, (Qnorm**2)*(npQ))



plt.title("1/(npQ*Q^2) for Coarse Grain")
plt.plot(npQinv30[0], npQinv30[1], 'o', label='30')
plt.plot(npQinv25[0], npQinv25[1], 'o', label='25')
plt.plot(npQinv22[0], npQinv22[1], 'o', label='22')
plt.plot(npQinv18[0], npQinv18[1], 'o', label='18')
plt.axhline(y = 2*B0, color='r', linestyle='-')
plt.ylim((0,2*2*B0))
plt.legend(loc="upper right")
plt.show()


plt.title("1/(npQ*Q^2) for Coarse Grain")
plt.plot(npQ30[0], npQ30[1], 'o', label='30')
plt.plot(npQ25[0], npQ25[1], 'o', label='25')
plt.plot(npQ22[0], npQ22[1], 'o', label='22')
plt.plot(npQ18[0], npQ18[1], 'o', label='18')
plt.legend(loc="upper right")
plt.show()

plt.plot(npqinv30[0], npqinv30[1], 'o', label='q')
plt.plot(npQinv30[0], npQinv30[1], 'o', label='Q')
plt.legend(loc="upper right")
plt.show()


plt.plot(npqinv25[0], npqinv25[1], 'o', label='q')
plt.plot(npQinv25[0], npQinv25[1], 'o', label='Q')
plt.legend(loc="upper right")
plt.show()


'''
       
    # plt.figure(3)
    plt.title("1/n_perp_q for Coarse Grain \n Loaded from \n" + file_name_load)
    plt.plot(qnorm, 1/nperp_q, 'o')
    plt.axhline(y = 2*Kt0, color='r', linestyle='-')
    #plt.ylim((0,2*B0))
    plt.show()
    
    # plt.figure(3)
    plt.title("1/n_perp_Q NO twist  for Coarse Grain \n Loaded from \n" + file_name_load)
    plt.plot(Qnorm, 1/nperp_Q, 'o')
    plt.axhline(y = 2*Kt0, color='r', linestyle='-')
    #plt.ylim((0,2*B0))
    plt.show()
    
    #plt.figure(4)
    plt.title("1/n_perp q twisted for Coarse Grain \n Loaded from \n" + file_name_load)
    plt.plot(qnorm, 1/(nperp_q*(1 + 0.06*qnorm**(2))), 'o')
    plt.axhline(y = 2*Kt0, color='r', linestyle='-')
    #plt.ylim((0,2*B0))
    plt.show()

    #plt.figure(4)
    plt.title("1/n_perp Q twisted for Coarse Grain \n Loaded from \n" + file_name_load)
    plt.plot(Qnorm, 1/(nperp_Q*(1 + 0.15*Qnorm**(2))), 'o')
    plt.axhline(y = 2*Kt0, color='r', linestyle='-')
    #plt.ylim((0,2*B0))
    plt.show()
    
'''
    
    # #plt.figure(4)
    # beta_twist_CG = 0.05
    # plt.title("1/n_perp Q with twist for Coarse Grain \n Loaded from \n" + file_name_load)
    # plt.plot(Qnorm, 1/(nperp_Q*(1 + beta_twist_CG*Qnorm**(2))), 'o')
    # plt.axhline(y = 2*Kt0, color='r', linestyle='-')
    # #plt.ylim((0,2*B0))
    # plt.show()
    
    # # #plt.figure(5)
    # # plt.title("1/Hq for Coarse Grain \n Loaded from \n" + file_name_load)
    # # plt.plot(Qnorm, (1 + (Qnorm**2)*B0/Kt0)/((Qnorm**4)*Hq), 'o')
    # # plt.axhline(y = B0, color='r', linestyle='-')
    # # plt.ylim((0,1.5*B0))
    # # plt.show()
    
    
    # plt.figure(1)
    # plt.title("1/np q for Coarse Grain \n Loaded from \n" + file_name_load)
    # plt.plot(qnorm, npq*(qnorm**2), 'o')
    # plt.axhline(y = B0, color='r', linestyle='-')
    # #plt.ylim((0, 3/B0))
    # plt.show()
    
'''    
#%%

#plt.figure(2)
plt.title("(npQ*Q^2) for Coarse Grain \n Loaded from \n" + file_name_load)
plt.plot(Qnorm, ((Qnorm**2)*npQ) + ((Qnorm**2)*0.01), 'o')
plt.show()

plt.title("npq*q^2 for Coarse Grain \n Loaded from \n" + file_name_load)
plt.plot(qnorm, npq*(qnorm**2), 'o')
plt.show()

#%%

#plt.figure(2)
plt.title("(nperpQ) for Coarse Grain \n Loaded from \n" + file_name_load)
plt.plot(Qnorm, nperp_Q, 'o')
plt.show()

plt.title("nperpq for Coarse Grain \n Loaded from \n" + file_name_load)
plt.plot(qnorm, nperp_q, 'o')
plt.show()

#%%

# max((np.linalg.norm(CG_frames, axis=3)).flatten())


plt.title("npq*q^2 for Coarse Grain \n Loaded from \n" + file_name_load)
plt.plot(qnorm, npq*(qnorm**2), 'o')
plt.show()


plt.title("npQ for Coarse Grain \n Loaded from \n" + file_name_load)
plt.plot(qnorm[20:], npq[20:], 'o')
#plt.ylim((0.035,0.045))
plt.show()



plt.title("npQ for Coarse Grain \n Loaded from \n" + file_name_load)
plt.plot(Qnorm[60:], (Qnorm[60:])**2, 'o')
#plt.ylim((0.035,0.045))
plt.show()

'''
#%%


#%%

from statsmodels.graphics.tsaplots import plot_acf
    

#dirx = CG_frames[:,:,:,0].flatten()
dirx = CG_frames[:,5,4,0]

bb = []

for nn in [6000, 16000, 36000, 56000, 90000, 96000]:

    aa = []
    dirx = CG_frames[nn,:,:,0].flatten()
    for i in range(3000):
    
        dirx2 = CG_frames[nn+i,:,:,0].flatten()
        corr= abs(np.corrcoef(dirx, dirx2)[0,1])
        aa = aa + [corr]
    bb = bb + [aa]
    
for i in range(5):
    plt.plot(list(range(3000)), bb[i])

plt.show()


plt.plot(list(range(3000)), bb[0])


np.corrcoef(dirx, dirx2)


plot_acf(dirx[:20000], lags=400)


2000/100000



# np.sort(dirx)

# np.mean(dirx)

# plt.hist(dirx,bins=100,range=(-0.5,0.5))
# plt.show()


# dir_slit = np.split(dirx[:5000], 100)

# stds = [np.std(dir_win) for dir_win in dir_slit]

# plt.plot(stds)




#%%
#MC_frames, en_lst, en2_lst = run_MC_2D(N_frames = 100, period = 10*N0*N0, delta = 0.05, warm = 50, N = N0, L = L0, B = B0, Kt = Kt0, do_another=True)
#max(abs(en_lst - en2_lst))

NN = 11
dx = 16/11
curv_lst = []
for n in MC_frames:

    for i in range(NN):
        for j in range(NN):
            curv = ((n[(i+1)%NN, j, 0] - n[i, j, 0])/dx + (n[i, (j+1)%NN, 1] - n[i, j, 1])/dx)
            curv_lst.append(curv)


plt.hist(curv_lst, bins = 150)
#, range=(-1.5,1.5)


#%%
