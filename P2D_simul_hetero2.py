"""
Simulation of P2D considering multiple conformations (Jongmin Sung)

"""

from __future__ import division, print_function, absolute_import
import numpy as np
import matplotlib.pyplot as plt
import scipy.special as sp
from scipy.optimize import minimize 
import random 
from scipy.stats import gamma
from scipy.stats import chisqprob

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
    result = minimize(fun, p0, method='SLSQP', args=(s, r)) 
    return result
    
# Parameters
mu = [5.0, 10.0]
s_m = 5.0 # sigma mean
s_s = s_m/3.0 # sigma sigma
N_i = [500, 500]
N = sum(N_i)
num_iter = 10*N
prob = 0.00001
bin1d = 50
bin2d = 20

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

# Generate data
s_i = np.random.gamma(s_shape, s_scale, N)
x_i = s_i * np.random.randn(N) + mu_i
y_i = s_i * np.random.randn(N)  
r_i = (x_i**2.0 + y_i**2.0)**0.5

# P2D MLE fitting 
result0 = MLE(mu_m, s_i, r_i); mu0 = result0["x"]; score0 = result0["fun"] 

################################################################################
# MC search for multiple conformation
################################################################################
#gg1 = np.array([0]*int(N/2))
#gg2 = np.array([1]*(N-int(N/2)))
#gg = np.concatenate((gg1, gg2))

gg = []
for i in range(len(r_i)):
    if r_i[i] < np.median(r_i):
        gg.append(0)
    else:
        gg.append(1)
gg = np.array(gg)
    
mu1 = [mu0]
mu2 = [mu0]
score1 = [score0]
score2 = [score0]
score12 = [score0]
accept = [0]
g_right = [sum(group==gg)]

for i in range(num_iter):
    pick1 = np.random.choice(np.where(gg==0)[0], size=1)[0]
    pick2 = np.random.choice(np.where(gg==1)[0], size=1)[0]

    gg_temp = gg.copy()
    gg_temp[pick1] = 1
    gg_temp[pick2] = 0
          
    r1_temp = r_i[gg_temp == 0]; s1_temp = s_i[gg_temp == 0]
    r2_temp = r_i[gg_temp == 1]; s2_temp = s_i[gg_temp == 1]
    
    result1 = MLE(mu1[-1], s1_temp, r1_temp); mu1_temp = result1["x"]; score1_temp = result1["fun"]
    result2 = MLE(mu2[-1], s2_temp, r2_temp); mu2_temp = result2["x"]; score2_temp = result2["fun"]

    n1 = sum(gg == 0)
    n2 = sum(gg == 1)
    n1_temp = sum(gg_temp == 0)
    n2_temp = sum(gg_temp == 1)


    score12_temp = score1_temp + score2_temp
    
    accept1 = score1_temp/n1_temp < score1[-1]/n1 
    accept2 = score2_temp/n2_temp < score2[-1]/n2 
   
    if (accept1 & accept2) :       
        gg = gg_temp.copy()
        accept.append(accept[-1]+1)
        mu1.append(mu1_temp)
        mu2.append(mu2_temp)
        score1.append(score1_temp)
        score2.append(score2_temp)
        score12.append(score12_temp)
    else:
        accept.append(accept[-1])
        mu1.append(mu1[-1])
        mu2.append(mu2[-1])    
        score1.append(score1[-1]) 
        score2.append(score2[-1]) 
        score12.append(score12[-1])    
  
    g_right_new = sum(group==gg)
    g_right.append(g_right_new)  
                                            
    if i%(num_iter/100) == 0:
        print(int(i/num_iter*100))

# p-value
p_value = chisqprob(2*(score0-score12[-1]), 2)

# If mu1 and mu2 are swapped
if mu1[-1] > mu2[-1]:
    mu_temp = mu1
    mu1 = mu2
    mu2 = mu_temp
    gg = 1 - gg

g_right_percent = np.array(g_right)/N*100
if g_right_percent[-1] < 50:
    g_right_percent = 100 - g_right_percent

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

################################################################################
# Figure 
################################################################################
plt.close('all')
fig1 = plt.figure(1)

# sp1. Mu distribution
sp1 = fig1.add_subplot(3,4,1)
sp1.hist(mu_i, bins=1000, normed=False, color='k', histtype='step', linewidth=2)
title1 = 'mu = %.1f (%d), %.1f (%d) (%.1f +/- %.1f) ' \
                % (mu[0], N_i[0], mu[1], N_i[1], mu_m, mu_s)
sp1.axis([0, mu[1]*1.2, 0, max(N_i)*1.2])
sp1.set_title(title1)
sp1.axvline(x=mu_m, color='k', linewidth=0.5)

# sp2. Sigma distribution
sp2 = fig1.add_subplot(3,4,2)
hist_sigma = sp2.hist(s_i, bins='scott', normed=False, color='k', histtype='step', linewidth=2)
nc_sigma = N*(hist_sigma[1][1] - hist_sigma[1][0])
x_sigma = np.linspace(0, max(s_i), 100)
sp2.plot(x_sigma, nc_sigma*gamma.pdf(x_sigma, s_shape, 0, s_scale), 'b', linewidth=2)
title2 = 'Sigma (%.1f +/- %.1f)' % (s_m, s_s)
sp2.set_title(title2)

# sp3. Euclidean distance - histogram
sp3 = fig1.add_subplot(3,4,3)
hist_ed = sp3.hist(r_i, bins='scott', normed=False, color='k', histtype='step', linewidth=2)
nc_ed = N*(hist_ed[1][1] - hist_ed[1][0])
x_ed = np.linspace(max(min(r_i), 0), max(r_i), 100)
sp3.set_title('Euclidean distance (R)')
sp3.axvline(x=mu[0], color='b', linewidth=2)
sp3.axvline(x=mu[1], color='r', linewidth=2)

# sp4. P2D MLE with mu
sp4 = fig1.add_subplot(3,4,4)
sp4.hist(r_i, bins='scott', normed=False, color='k', histtype='step', linewidth=2)
P2D0 = np.zeros((N, len(x_ed)), dtype=float)
for i in range(len(s_i)):
    P2D0[i] = P2D(mu0, s_i[i], x_ed)
P2D0m = np.mean(P2D0, axis=0)
sp4.plot(x_ed, nc_ed*P2D0m, 'r', linewidth=2)
title4 = 'mu = %.1f +/- %.1f\nLL = %d'  % (mu0, mu_s0, score0)
sp4.set_title(title4)

# sp5. P2D MLE with mu1 and mu2
sp5 = fig1.add_subplot(3,4,5)
sp5.hist(r_i, bins='scott', normed=False, color='k', histtype='step', linewidth=2)
P2D1 = np.zeros((N, len(x_ed)), dtype=float)
for i in range(len(s_i)):
    if gg[i] == 0:
        P2D1[i] = P2D(mu1[-1], s_i[i], x_ed)
    else:
        P2D1[i] = P2D(mu2[-1], s_i[i], x_ed)                
P2D1m = np.mean(P2D1, axis=0)
sp5.plot(x_ed, nc_ed*P2D1m, 'r', linewidth=2)
title5 = 'mu1 = %.1f +/- %.1f (%d), mu2 = %.1f +/- %.1f (%d) \n\
        LL = %d (p-value = %e)'  \
            % (mu1[-1], mu_s1, N-sum(gg), mu2[-1], mu_s2, sum(gg), score12[-1], p_value)
sp5.set_title(title5)

# sp6, sp7. mu1 / mu2
sp6 = fig1.add_subplot(3,4,6)
sp7 = fig1.add_subplot(3,4,7)
hist_1 = sp6.hist(r_i[group==0], bins='scott', normed=False, color='k', histtype='step', linewidth=2)
hist_2 = sp7.hist(r_i[group==1], bins='scott', normed=False, color='k', histtype='step', linewidth=2)
nc_1 = (N-sum(group))*(hist_1[1][1] - hist_1[1][0])
nc_2 = (sum(group))*(hist_2[1][1] - hist_2[1][0])
P2D_1 = np.zeros((N-sum(group), len(x_ed)), dtype=float)
P2D_2 = np.zeros((sum(group), len(x_ed)), dtype=float)
j = 0
k = 0
for i in range(N):
    if group[i] == 0:
        P2D_1[j] = P2D(mu1[-1], s_i[i], x_ed); j += 1
    else:
        P2D_2[k] = P2D(mu2[-1], s_i[i], x_ed); k += 1                
P2D_1m = np.mean(P2D_1, axis=0)
P2D_2m = np.mean(P2D_2, axis=0)
sp6.plot(x_ed, nc_1*P2D_1m, 'r', linewidth=2); sp6.set_title('mu1')
sp7.plot(x_ed, nc_2*P2D_2m, 'r', linewidth=2); sp7.set_title('mu2')

# sp8. mu1/m2 iteration
sp8 = fig1.add_subplot(3,4,8)
sp8.plot(mu1, 'b', mu2, 'r')
sp8.axhline(y=mu[0], color='k', linewidth=0.5)
sp8.axhline(y=mu[1], color='k', linewidth=0.5)
sp8.axis([0, len(mu1), 0, max(max(mu1), max(mu2))*1.2])
title8 = 'mu1 = %.1f +/- %.1f, mu2 = %.1f +/- %.1f' \
        % (mu1[-1], mu_s1, mu2[-1], mu_s2)
sp8.set_title(title8)

# sp9. score iteration
sp9 = fig1.add_subplot(3,4,9)
sp9.plot(score12[1:], 'k')
sp9.set_title('LogLikelihood')

# sp10. group difference
sp10 = fig1.add_subplot(3,4,10)
sp10.plot(g_right_percent, 'k')
title10 = 'Correct Group = %.1f %%' % (g_right_percent[-1])
sp10.set_title(title10)

# sp11. accept iteration
sp11 = fig1.add_subplot(3,4,11)
sp11.plot(np.array(accept), 'k')
sp11.set_title('Cumulative acceptance')

# sp12.score difference
sp12 = fig1.add_subplot(3,4,12)
score_diff = np.array(score12[:-1]) - np.array(score12[1:]) 
sp12.semilogy(score_diff, 'k.')
sp12.set_title('Score difference')

plt.subplots_adjust(wspace=0.3, hspace=0.5)
plt.show()



"""
# Subplots used previously (maybe useful again later)

# sp3. 2D localization histogram
sp3 = fig1.add_subplot(3,3,3)
sp3.hist2d(x_i, y_i, bins=bin2d)
sp3.set_title('2D Localization')
sp3.set_aspect('equal')
sp3.axhline(y=0, color='w', linewidth=0.5)
sp3.axvline(x=0, color='w', linewidth=0.5)


# sp3. R vs sigma 2D histogram
sp3 = fig1.add_subplot(3,3,3)
sp3.hist2d(r_i, s_i, bins=bin2d)
sp3.set_aspect('equal')
sp3.axhline(y=s_m, color='w', linewidth=0.5)
title3 = 'R vs Sigma (corr = %.2f)' % (np.corrcoef(r_i, s_i)[0,1])
sp3.set_title(title3)
plt.xlabel('Euclidean distance')
plt.ylabel('Sigma')

# sp7. LogLikelihood plot
sp8 = fig1.add_subplot(3,4,8)
sp8.plot(mu_range, LL_si, 'k-')
plt.title('LogLikelihood')

"""



