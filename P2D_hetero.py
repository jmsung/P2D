"""
Simulation and analysis of P2D with multiple conformations (Jongmin Sung)

class Sample()
- N_mol, mu_m, mu_s, s_m, s_s, generate(), mols = [Mol()], analyze(), mu_estimate

class Mol()
- mu, sigma, x, y, r

P2D(), LL(), MLE()

"""

from __future__ import division, print_function, absolute_import
import numpy as np
import matplotlib.pyplot as plt
import scipy.special as sp
from scipy.optimize import minimize 
from scipy.stats import gamma
from scipy.stats import chisqprob
import scipy.integrate as integrate
import random 
import time

pi = 3.141592
num_min = 1e-100
num_bin = 50

def LL(m, s, r):
    "LogLikelihood "   
    LL = np.sum(np.log10(P2D(m, s, r)))
    return LL
    
def MLE(m, s, r): 
    "P2D MLE with fixed mean sigma"
    fun = lambda *args: -LL(*args)
    p0 = [m]
    result = minimize(fun, p0, method='SLSQP', args=(s, r)) 
    return result


def P2D(m, s, r): 
    """P2D(r|m,s) = (r/s^2)*exp(-(m^2+r^2)/(2*s^2))*I0(r*m/s^2) Eq. (4)
    where, I0 = Modified Bessel function of order 0. """   
    P2D = (r/s**2)*np.exp(-(m**2 + r**2)/(2*s**2))*sp.i0(r*m/s**2)  
    P2D[np.isnan(P2D)] = num_min
    P2D[np.isinf(P2D)] = num_min
    P2D[P2D < num_min] = num_min
    return P2D 
    
def P2D_approx(m, s, r): 
    "Approximation of P2D when m >> s. "   
    P2D = (r/(2*pi*s*m))**0.5 * np.exp(-(r-m)**2/(2*s**2))
    iNaNs = np.isnan(P2D)
    P2D[iNaNs] = num_min
    P2D[P2D < num_min] = num_min
    return P2D   
    
def P2D_generate(m, s, N):
    "Generate N random variable of mu from P2D distribution"
    r = np.linspace(max(num_min, m-10*s), m+10*s, num=100000)
    if m < 10*s:
        pdf = P2D(m, s, r)
    else:
        pdf = P2D_approx(m, s, r)
    pdf = pdf/sum(pdf)
    return np.random.choice(r, size=N, p=pdf).tolist()
    

def sigma_generate(m, s, N):
    "Generate N random variable of sigma from gamma distribution"
    shape = (m/s)**2.0
    scale = (s**2.0)/m
    return np.random.gamma(shape, scale, size=N).tolist()
                
# Class Mol 
class Mol(object):
    def __init__(self, m, s, i):
        self.group_num = i  
        self.m = m
        self.s = s   
        self.x0 = m
        self.y0 = 0        
        self.x = np.random.normal(self.x0, self.s)
        self.y = np.random.normal(self.y0, self.s)        
        self.r = (self.x**2.0 + self.y**2.0)**0.5

                                                
# Class Sample with multiple molecules    
class Sample(object):
    def __init__(self, groups):
        self.N_group = len(groups)
        
        self.group = [None]*self.N_group
        for i in range(self.N_group):
            self.group[i] = {
                'N' : groups[i]['N'],
                'm_m' : groups[i]['m_m'],
                'm_s' : groups[i]['m_s'],
                's_m' : groups[i]['s_m'],
                's_s' : groups[i]['s_s'],
                'c' : groups[i]['c'],
                'm' : [],
                's' : [],                
                'x' : [],
                'y' : [],
                'r' : []               
            }
                  
    def generate_data(self): 
        self.g = []
        self.mols = []      
        self.m = []
        self.s = []
        self.x = []
        self.y = []
        self.r = []
        
        for i in range(self.N_group):
            g = self.group[i]      
            g['m'] = P2D_generate(g['m_m'], g['m_s'], g['N'])    
            g['s'] = sigma_generate(g['s_m'], g['s_s'], g['N']) 

            # Generate N molecules with parameters 
            for j in range(g['N']):
                mol = Mol(g['m'][j], g['s'][j], i)
                self.mols.append(mol)
                g['x'] += [mol.x]
                g['y'] += [mol.y]
                g['r'] += [mol.r]    
                   
            self.g += [i]         
            self.m += g['m']
            self.s += g['s']
            self.x += g['x']
            self.y += g['y']
            self.r += g['r']          
                         
        self.m_max, self.m_min = max(self.m), min(self.m)
        self.s_max, self.s_min = max(self.s), min(self.s)
        self.x_max, self.x_min = max(self.x), min(self.x)
        self.y_max, self.y_min = max(self.y), min(self.y)        
        self.r_max, self.r_min = max(self.r), min(self.r)                                       
                                                                                                               
    def plot_result(self):
        plt.close('all')
        self.fig1 = plt.figure(1)
        
        # Distribution of mu
        sp = self.fig1.add_subplot(231) 
        bins = np.linspace(self.m_min, self.m_max, num_bin)
        for i in range(self.N_group):   
            g = self.group[i]
            sp.hist(g['m'], bins, normed=False, color=g['c'], histtype='step', linewidth=1)
        sp.hist(self.m, bins, normed=False, color='k', histtype='step', linewidth=1)
        sp.set_title('Mu')
                      
        # Distribution of sigma
        sp = self.fig1.add_subplot(232) 
        bins = np.linspace(self.s_min, self.s_max, num_bin)
        for i in range(self.N_group):   
            g = self.group[i]
            sp.hist(g['s'], bins, normed=False, color=g['c'], histtype='step', linewidth=1)
        sp.hist(self.s, bins, normed=False, color='k', histtype='step', linewidth=1)
        sp.set_title('Sigma')

        # Distributio of r
        sp = self.fig1.add_subplot(233)
        bins = np.linspace(self.r_min, self.r_max, num_bin)
        for i in range(self.N_group):   
            g = self.group[i]
            sp.hist(g['r'], bins, normed=False, color=g['c'], histtype='step', linewidth=1)
        sp.hist(self.r, bins, normed=False, color='k', histtype='step', linewidth=1)
        sp.set_title('Euclidean distance')
        
        # Estimation of Mu
        sp = self.fig1.add_subplot(234)
        bins = np.linspace(self.r_min, self.r_max, num_bin)
        sp.hist(self.m, bins, normed=False, color='k', histtype='step', linewidth=1)  
        sp.hist(self.m_guess, bins, normed=False, color='r', histtype='step', linewidth=1)    
        sp.set_title('Mu estimation')        
        
        # Distribution of mu
        sp = self.fig1.add_subplot(235)
        sp.plot(self.LL_iter[1:], 'k')
        sp.set_title('LL')
        
        sp = self.fig1.add_subplot(236)
        sp.plot(self.dm, 'k')
        i_min = np.argmin(self.dm)
        sp.plot(i_min, self.dm[i_min], 'r.', markersize = 20)     
        sp.set_title('Mu difference')
        
        self.fig1.tight_layout()
        self.fig1.subplots_adjust(wspace=0.1, hspace=0.3)       
           
                      
                                            
    def analyze_data(self):
        self.n_data = len(self.m)
        self.n_iter = self.n_data*10

        m_bin = np.linspace(self.r_min+1, self.r_max, num_bin)
        self.m_guess = np.random.choice(m_bin, size=self.n_data)           
        LL_iter = [1e100] 
        dm = []
  
        start = time.clock()
        for i in range(self.n_iter):
            m_temp = self.m_guess.copy()
            m_temp[i%self.n_data] = np.random.choice(m_bin[1:], size=1) 
            LL_temp = -LL(m_temp, np.array(self.s), np.array(self.r))/self.n_data
            if LL_temp < LL_iter[-1]:
                LL_iter += [LL_temp]
                self.m_guess = m_temp
            else:
                LL_iter += [LL_iter[-1]]
            dm += [np.sum(np.abs(np.array(self.m_guess) - np.array(self.m)))/self.n_data]
            
            done = (i+1)/self.n_iter*100
            if done%10 == 0:
                now = time.clock()
                spent = (now-start)/60 # time passed in min
                print('%d %%, %.1f min' %(done, (spent*(100-done)/done)))
                
        self.LL_iter = np.array(LL_iter)
        self.dm = np.array(dm)
                
   

# Parameters [N, m_m, m_s, s_m, s_s, c]
# N: Number of particle
# m_m: mu, mean
# m_s: mu, std
# s_m: sigma, mean
# s_m: sigma, std
# c: color

g1 = {'N':2000, 'm_m':10, 'm_s':5, 's_m':1, 's_s':0.1, 'c':'b'}
g2 = {'N':2000, 'm_m':20, 'm_s':5, 's_m':1, 's_s':0.1, 'c':'r'}
groups = [g1, g2]

sample = Sample(groups)
sample.generate_data()
sample.analyze_data()
sample.plot_result()

plt.show()


"""
        # Scatter plot of the groups
        sp = self.fig1.add_subplot(333) 
        for i in range(self.N_group):   
            g = self.group[i]
            c = g['c']
            sp.plot(g['x'], g['y'], c+'.', markersize=1, alpha=0.5)
        sp.axvline(x=0, color='k', linewidth=0.5)
        sp.axhline(y=0, color='k', linewidth=0.5)
        sp.set_aspect('equal')

        # Distribution in x
        sp = self.fig1.add_subplot(334)
        bins = np.linspace(self.x_min, self.x_max, num_bin)
        for i in range(self.N_group):   
            g = self.group[i]
            sp.hist(g['x'], bins, normed=False, color=g['c'], histtype='step', linewidth=1)
        sp.hist(self.x, bins, normed=False, color='k', histtype='step', linewidth=1)
        sp.set_title('X localization')        

        # Distribution in y
        sp = self.fig1.add_subplot(335)
        bins = np.linspace(self.y_min, self.y_max, num_bin)
        for i in range(self.N_group):   
            g = self.group[i]
            sp.hist(g['y'], bins, normed=False, color=g['c'], histtype='step', linewidth=1)
        sp.hist(self.y, bins, normed=False, color='k', histtype='step', linewidth=1)
        sp.set_title('Y localization')   
"""    