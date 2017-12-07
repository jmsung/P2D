"""
Simulation of P2D considering multiple conformations (Jongmin Sung)

"""

from __future__ import division, print_function, absolute_import
import numpy as np
import matplotlib.pyplot as plt
import scipy.special as sp
from scipy.optimize import minimize 
import random 

# P2D(r|mu,s) = (r/s^2)*exp(-(mu^2+r^2)/(2*s^2))*I0(r*mu/s^2) Eq. (4)
# I0 = Modified Bessel function of order 0.

def P2D(mu, s, r):   
    return (r/s**2)*np.exp(-(mu**2 + r**2)/(2*s**2))*sp.i0(r*mu/s**2)

# LogLikelihood 
def LL(param, s, r):
    mu = np.abs(param)      
    return np.sum(np.log10(P2D(mu, s, r)))

def MLE(mu, s, r): # P2D MLE with fixed mean sigma
    fun = lambda *args: -LL(*args)
    p0 = [mu]
    bnds = [(0.001, 10*mu)]
    result = minimize(fun, p0, method='SLSQP', bounds=bnds, args=(s, r)) 
    return result
    
# Parameters
mu = [8.0, 10.0]
N_i = [2000, 1000]
N = sum(N_i)
num_iter = 10*N
s_m = 2.0 # sigma mean
s_s = s_m/3.0 # sigma sigma

group1 = np.array([0]*N_i[0])
group2 = np.array([1]*N_i[1])
group = np.concatenate((group1, group2))
random.shuffle(group)

mu_i = np.zeros(N)
for i in range(N):
    mu_i[i] = mu[group[i]]

mu_m = np.mean(mu_i)
mu_s = np.std(mu_i)
mu_range = np.linspace(min(mu)-2, max(mu)+2, 100)

s_shape = (s_m/s_s)**2.0
s_scale = (s_s**2.0)/s_m
s_i = np.random.gamma(s_shape, s_scale, N)

x_i = s_i * np.random.randn(N) + mu_i
y_i = s_i * np.random.randn(N)  
r_i = (x_i**2.0 + y_i**2.0)**0.5

# P2D MLE fitting 
result0 = MLE(mu_m, s_i, r_i); mu0 = result0["x"]; score0 = result0["fun"] 

# MC search for multi conformation

gg1 = np.array([0]*int(N/2))
gg2 = np.array([1]*(N-int(N/2)))
gg = np.concatenate((gg1, gg2))
mu1 = [mu0]
mu2 = [mu0]
score = [score0]
accept = [0]
g_diff = [sum(abs(group-gg))]

for i in range(num_iter):
    pick = np.random.randint(N)
    gg_temp = gg.copy()
    if gg_temp[pick] == 0: 
        gg_temp[pick] = 1
    else: 
        gg_temp[pick] = 0
    
    r1 = r_i[gg_temp == 0]; s1 = s_i[gg_temp == 0]
    r2 = r_i[gg_temp == 1]; s2 = s_i[gg_temp == 1]
    
    result1 = MLE(mu1[-1], s1, r1); mu1_temp = result1["x"]; score1 = result1["fun"]
    result2 = MLE(mu2[-1], s2, r2); mu2_temp = result2["x"]; score2 = result2["fun"]
    score_temp = score1 + score2
   
    score_diff = score_temp - score[-1]
   
    if score_diff < 0.0:
        gg = gg_temp.copy()
        accept.append(accept[-1]+1)
        mu1.append(mu1_temp)
        mu2.append(mu2_temp)
        score.append(score_temp)
    else:
        accept.append(accept[-1])
        mu1.append(mu1[-1])
        mu2.append(mu2[-1])    
        score.append(score[-1])    
        
    g_diff_new = sum(abs(group-gg))
    g_diff.append(g_diff_new)  
                                            
    if i%(num_iter/100) == 0:
        print(int(i/num_iter*100))


# If mu1 and mu2 are swapped
if mu1[-1] > mu2[-1]:
    mu_temp = mu1
    mu1 = mu2
    mu2 = mu_temp
    gg = 1 - gg

g_diff_percent = np.array(g_diff)/g_diff[0]*100
if g_diff_percent[-1] > 100:
    g_diff_percent = 200 - g_diff_percent

# LogLikelihood surface 
LL_si = np.zeros(len(mu_range))
i = 0
for mu_scan in mu_range:  
    LL_si[i] = abs(LL(mu_scan, s_i, r_i));     
    i+=1

# Error estimation
dmu = 0.001
Info0 = abs(LL(mu0+dmu, s_i, r_i) + LL(mu0-dmu, s_i, r_i) - 2*LL(mu0, s_i, r_i))/dmu**2
mu_s0 = 1/Info0**0.5

s1 = s_i[gg==0]; r1 = r_i[gg==0]
s2 = s_i[gg==1]; r2 = r_i[gg==1]
Info1 = abs(LL(mu1[-1]+dmu, s1, r1) + LL(mu1[-1]-dmu, s1, r1) - 2*LL(mu1[-1], s1, r1))/dmu**2
Info2 = abs(LL(mu2[-1]+dmu, s2, r2) + LL(mu2[-1]-dmu, s2, r2) - 2*LL(mu2[-1], s2, r2))/dmu**2
mu_s1 = 1/Info1**0.5
mu_s2 = 1/Info2**0.5


# Figure 1
plt.close('all')
fig1 = plt.figure(1)
bin2d = 20

# sp1. Mu distribution
sp1 = fig1.add_subplot(3,4,1)
sp1.hist(mu_i, bins=1000, normed=False, color='k', histtype='step', linewidth=2)
title1 = 'Mu (%.1f +/- %.1f) = %.1f (%d), %.1f (%d)' \
                % (mu_m, mu_s, mu[0], N_i[0], mu[1], N_i[1])
sp1.set_title(title1)
sp1.axvline(x=mu_m, color='k', linewidth=0.5)

# sp2. Sigma distribution
sp2 = fig1.add_subplot(3,4,2)
sp2.hist(s_i, bins='scott', normed=False, color='k', histtype='step', linewidth=2)
title2 = 'Sigma (%.1f +/- %.1f)' % (s_m, s_s)
sp2.set_title(title2)
sp2.axvline(x=s_m, color='k', linewidth=0.5)

# sp3. 2D localization histogram
sp3 = fig1.add_subplot(3,4,3)
sp3.hist2d(x_i, y_i, bins=bin2d)
sp3.set_title('2D Localization')
sp3.set_aspect('equal')
sp3.axhline(y=0, color='w', linewidth=0.5)
sp3.axvline(x=0, color='w', linewidth=0.5)

# sp4. R vs sigma 2D histogram
sp4 = fig1.add_subplot(3,4,4)
sp4.hist2d(r_i, s_i, bins=bin2d)
sp4.set_aspect('equal')
sp4.axhline(y=s_m, color='w', linewidth=0.5)
title4 = 'R vs Sigma (corr = %.2f)' % (np.corrcoef(r_i, s_i)[0,1])
sp4.set_title(title4)
plt.xlabel('Euclidean distance')
plt.ylabel('Sigma')

# sp5. Euclidean distance - histogram
sp5 = fig1.add_subplot(3,4,5)
histxx = sp5.hist(r_i, bins='scott', normed=False, color='k', histtype='step', linewidth=2)
nc = N*(histxx[1][1] - histxx[1][0])
xx = np.linspace(max(min(r_i), 0), max(r_i), 100)
sp5.set_title('Euclidean distance (R)')
sp5.axvline(x=mu_m, color='k', linewidth=0.5)

# sp6. P2D MLE with mu
sp6 = fig1.add_subplot(3,4,6)
sp6.hist(r_i, bins='scott', normed=False, color='k', histtype='step', linewidth=2)
P2D0 = np.zeros((N, len(xx)), dtype=float)
for i in range(len(s_i)):
    P2D0[i] = P2D(mu0, s_i[i], xx)
P2D0m = np.mean(P2D0, axis=0)
sp6.plot(xx, nc*P2D0m, 'r', linewidth=2)
title6 = 'P2D (mu=%.1f+/-%.1f, LL=%d)'  % (mu0, mu_s0, score0)
sp6.set_title(title6)

# sp7. P2D MLE with mu1 and mu2
sp7 = fig1.add_subplot(3,4,7)
sp7.hist(r_i, bins='scott', normed=False, color='k', histtype='step', linewidth=2)
P2D1 = np.zeros((N, len(xx)), dtype=float)
for i in range(len(s_i)):
    if gg[i] == 0:
        P2D1[i] = P2D(mu1[-1], s_i[i], xx)
    else:
        P2D1[i] = P2D(mu2[-1], s_i[i], xx)                
P2D1m = np.mean(P2D1, axis=0)
sp7.plot(xx, nc*P2D1m, 'r', linewidth=2)
title7 = 'P2D (mu1=%.1f+/-%.1f, mu1=%.1f+/-%.1f, LL=%d)'  \
            % (mu1[-1], mu_s1, mu2[-1], mu_s2, score[-1])
sp7.set_title(title7)

# sp8. score iteration
sp8 = fig1.add_subplot(3,4,8)
sp8.plot(score, 'k')
sp8.set_title('LogLikelihood')

# sp9. mu1/m2 iteration
sp9 = fig1.add_subplot(3,4,9)
sp9.plot(mu1, 'r', mu2, 'b')
sp9.axhline(y=mu[0], color='k', linewidth=0.5)
sp9.axhline(y=mu[1], color='k', linewidth=0.5)
title9 = 'mu1 = %.1f +/- %.1f, mu2 = %.1f +/- %.1f' \
        % (mu1[-1], mu_s1, mu2[-1], mu_s2)
sp9.set_title(title9)

# sp10. group difference
sp10 = fig1.add_subplot(3,4,10)
sp10.plot(g_diff_percent, 'k')
sp10.axis([0, len(g_diff_percent), 0, 100])
title10 = 'Group difference = %.1f %%' % (g_diff_percent[-1])
sp10.set_title(title10)

# sp11. accept iteration
sp11 = fig1.add_subplot(3,4,11)
sp11.plot(np.array(accept)/num_iter*100, 'k')
sp11.set_title('Cumulative acceptance (%)')

# sp12.score difference
sp12 = fig1.add_subplot(3,4,12)
score_diff = np.array(score[:-1]) - np.array(score[1:]) 
sp12.semilogy(score_diff, 'k.')
sp12.set_title('Score difference')


plt.subplots_adjust(wspace=0.3, hspace=0.3)
plt.show()

