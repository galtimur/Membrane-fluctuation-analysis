#%%

import numpy as np
from MC_2D_softening_run import run_MC_2D
from fluctuation_analysis_2D_continuous import analyze_2D_fluct, bin_average
import matplotlib.pyplot as plt

#%%


L = 16
B0 = 10
Kt0 = 10
Ktw0 = 0
dx = 0.8
beta_twist = Ktw0/Kt0

#N_list = [7, 11, 15, 20]
N_list = [11]
#N_frames0 = 100

#%%

file_name = "D:\Timur\MD_Fluctuations\Results\CG_frames_L=366_all_lipids_5000_frames.npy"
frames = np.load(file_name)
frames_test = frames[:1000]
frames = np.load(file_name)
qnorm, npq, nperp_q, Hq = analyze_2D_fluct(frames_test, L, N=20, dx=dx)

#%%

#plt.plot(qnorm, 1/(npq*qnorm*qnorm), 'o')
#plt.ylim([0, 200])
plt.plot(qnorm, npq, 'o')
plt.show()

#%%


### импортируем данные на сетке и переводим их в формат, соответствующий формату для непррывного фурье (N_frames, N_lipids, 2, 3)

file_name = 'D:\Timur\MD_Fluctuations\Results\CG_frames_L=366_N=25.npy'
frames = np.load(file_name)
frames = frames[::100]

L = 36.6
N = 25
dx = L/N

frames_new = np.zeros((len(frames), N*N, 2, 3))

for nf, frame in enumerate(frames):
    count = 0
    for iy in range(N):
        for ix in range(N):
            nx, ny = dx*frame[ix, iy]
            x, y = [dx*ix, dx*iy]
            frames_new[nf, count] = np.array([[nx, ny, 0], [x, y, 0]])
            count += 1

#%%

qnorm, npq, nperp_q, Hq = analyze_2D_fluct(frames_new, L = L, N=25, dx=dx)

#%%

plt.plot(qnorm, 1/(npq*qnorm*qnorm), 'o')
plt.ylim([0, 20])
plt.show()

#%%

Do_calc = True
Do_analysis = False

if Do_calc:

    for N0 in N_list:
        
        # file_name = 'Results/MC_frames_' + str(int(N_frames0/1000)) + \
        #     'k_Kt=' + str(int(Kt0)) + \
        #     '_B=' + str(int(B0)) + \
        #     '_Ktw=' + str(int(Ktw0)) + \
        #     '_L=' + str(int(L0)) + \
        #     '_N=' + str(int(N0))
        
        file_name = 'Results/MC_frames_' + str(int(N_frames0/1000)) + \
            'k_direct_soft_en' + \
            '_B=' + str(int(B0)) + \
            '_N=' + str(int(N0))
        
        MC_frames = run_MC_2D(N_frames = N_frames0, period = 25*N0*N0, delta = 0.05, warm = 50, soft = False, N = N0, L = L0, B = B0, Kt = Kt0, Ktw = Ktw0)
        np.save(file_name, MC_frames)


if Do_analysis:

    results = []
    results_av = []
    for N0 in N_list:
        
        file_name = 'Results/MC_frames_' + str(int(N_frames0/1000)) + \
            'k_Kt=' + str(int(Kt0)) + \
            '_B=' + str(int(B0)) + \
            '_Ktw=' + str(int(Ktw0)) + \
            '_L=' + str(int(L0)) + \
            '_N=' + str(int(N0)) + '.npy'
        
        frames = np.load(file_name)
        qnorm, npq, Qnorm, npQ, nperp_q, nperp_Q, Hq, HQ = analyze_2D_fluct(frames, L = L0)

        qnorm_av, npq_av = bin_average(qnorm, npq, 0.2)
        Qnorm_av, npQ_av = bin_average(Qnorm, npQ, 0.2)
        
        qnorm_av, nperp_q_av = bin_average(qnorm, nperp_q, 0.2)
        Qnorm_av, nperp_Q_av = bin_average(Qnorm, nperp_Q, 0.2)
        
        qnorm_av, Hq_av = bin_average(qnorm, Hq, 0.2)
        Qnorm_av, HQ_av = bin_average(Qnorm, HQ, 0.2)
        
        results = results + [[qnorm, npq, Qnorm, npQ, nperp_q, nperp_Q, Hq, HQ]]
        results_av = results_av + [[qnorm_av, npq_av, Qnorm_av, npQ_av, nperp_q_av, nperp_Q_av, Hq_av, HQ_av]]
        print(N0)

#%%

def plot_res(results):

    file_name = 'MC_frames_' + str(int(N_frames0/1000)) + \
        'k_Kt=' + str(int(Kt0)) + \
        '_B=' + str(int(B0)) + \
        '_Ktw=' + str(int(Ktw0)) + \
        '_L=' + str(int(L0))
    
    
    plt.title("1/(q^2*np^2) q for Mone-Carlo \n Loaded from \n" + file_name)
    plt.axhline(y = B0, color='r', linestyle='-')
    plt.ylim((0,1.5*B0))
    
    for res in results:
    
        qnorm, npq, nperp_q, Hq = res
        plt.plot(Qnorm, 1/((Qnorm**2)*npQ), 'o')
    
    plt.legend(['B0'] + N_list, loc='lower right')
    plt.show()
    
    
    plt.title("1/(nperp_Q^2*(1 + beta_twist*Qnorm**(2))) Q \n for Mone-Carlo \n Loaded from \n" + file_name)
    plt.axhline(y = Kt0, color='r', linestyle='-')
    plt.ylim((0,1.5*B0))
    
    for res in results:
    
        qnorm, npq, Qnorm, npQ, nperp_q, nperp_Q, Hq, HQ = res
        plt.plot(Qnorm, 1/(nperp_Q*(1 + beta_twist*Qnorm**(2))), 'o')
    
    plt.legend(['Kt0'] + N_list, loc='lower right')
    plt.show()   
    
    
    plt.title("1/Hq^2 Q for Mone-Carlo \n Loaded from \n" + file_name)
    plt.axhline(y = Kt0, color='r', linestyle='-')
    plt.ylim((0,1.5*B0))
    
    for res in results:
    
        qnorm, npq, Qnorm, npQ, nperp_q, nperp_Q, Hq, HQ = res
        plt.plot(Qnorm, (1 + (Qnorm**2)*B0/Kt0)/((Qnorm**4)*HQ), 'o')
    
    plt.legend(['Kt0'] + N_list, loc='lower right')
    plt.show() 
    
    
    plt.title("(Q^2*np^2) for Mone-Carlo \n Loaded from \n" + file_name)
    plt.axhline(y = 1/B0, color='r', linestyle='-')
    plt.ylim((0,1.5/B0))
    
    for res in results:
    
        qnorm, npq, Qnorm, npQ, nperp_q, nperp_Q, Hq, HQ = res
        plt.plot(Qnorm, ((Qnorm**2)*npQ), 'o')
    
    plt.legend(['1/B0'] + N_list, loc='lower right')
    plt.show()
    
    
    plt.title("(q^2*np^2) for Mone-Carlo \n Loaded from \n" + file_name)
    plt.axhline(y = 1/B0, color='r', linestyle='-')
    
    for res in results:
    
        qnorm, npq, Qnorm, npQ, nperp_q, nperp_Q, Hq, HQ = res
        plt.plot(qnorm, ((qnorm**2)*npq), 'o')
    
    plt.legend(['1/B0'] + N_list, loc='lower right')
    plt.show()
    
    
    plt.title("(HQ^2 - 1/Q^2/Kt0)*(Q^4) \n Q for Mone-Carlo \n Loaded from \n" + file_name)
    plt.axhline(y = 1/B0, color='r', linestyle='-')
    plt.ylim((0,1.5/B0))
    
    for res in results:
    
        qnorm, npq, Qnorm, npQ, nperp_q, nperp_Q, Hq, HQ = res
        plt.plot(Qnorm, (HQ - 1/Qnorm**2/Kt0)*(Qnorm**4), 'o')
    
    plt.legend(['1/Kt0'] + N_list, loc='lower right')
    plt.show()
    
    plt.title("1/q^2/Kt0)*(q^4) \n 1/Hq q for Mone-Carlo \n Loaded from \n" + file_name)
    plt.axhline(y = 1/B0, color='r', linestyle='-')
    plt.ylim((0,20/B0))
    
    for res in results:
    
        qnorm, npq, Qnorm, npQ, nperp_q, nperp_Q, Hq, HQ = res
        plt.plot(qnorm, (Hq - 1/qnorm**2/Kt0)*(qnorm**4), 'o')
    
    plt.legend(['1/Kt0'] + N_list, loc='lower right')
    plt.show()
    
    plt.title("nperp_Q^2*(1 + beta_twist*Qnorm**(2))) Q \n for Mone-Carlo \n Loaded from \n" + file_name)
    plt.axhline(y = 1/Kt0, color='r', linestyle='-')
    plt.ylim((0,1.5/Kt0))
    
    for res in results:
    
        qnorm, npq, Qnorm, npQ, nperp_q, nperp_Q, Hq, HQ = res
        plt.plot(Qnorm, (nperp_Q*(1 + beta_twist*Qnorm**(2))), 'o')
    
    plt.legend(['Kt0'] + N_list, loc='lower right')
    plt.show()
    
    plt.title("nperp_q^2*(1 + beta_twist*qnorm**(2))) q \n for Mone-Carlo \n Loaded from \n" + file_name)
    plt.axhline(y = 1/Kt0, color='r', linestyle='-')
    plt.ylim((0,1.5/Kt0))
    
    for res in results:
    
        qnorm, npq, Qnorm, npQ, nperp_q, nperp_Q, Hq, HQ = res
        plt.plot(qnorm, (nperp_q*(1 + beta_twist*qnorm**(2))), 'o')
    
    plt.legend(['Kt0'] + N_list, loc='lower right')
    plt.show()

#%%

plot_res(results)

#%%

test_n = nperp_q*(1 + beta_twist*qnorm**(2))
qnorm

bin_centers, bin_means = bin_average(qnorm, test_n, 0.2)
plt.plot(bin_centers, bin_means, 'o')

#%%





#plt.figure(0)
plt.title("np q for Mone-Carlo \n Loaded from \n" + file_name)
plt.axhline(y = 1/B0, color='r', linestyle='-')
plt.ylim((0, 1.5*1/B0))

for res in results:

    qnorm, npq, Qnorm, npQ, nperp_q, nperp_Q = res
    plt.plot(qnorm, npq, 'o', label = 'wewfv')

plt.legend(['B0'] + N_list, loc='lower right')
plt.show()

#%%

