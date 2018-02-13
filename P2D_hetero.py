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
num_min = 1e-300
num_bin = 100

def LL(m, s, r):
    "LogLikelihood "   
    return np.sum(np.log10(P2D(m, s, r)))

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
    iNaNs = np.isnan(P2D)
    P2D[iNaNs] = num_min
    P2D[P2D < num_min] = num_min
    return P2D 
    
def P2D_approx(m, s, r): 
    P2D = (r/(2*pi*s*m))**0.5 * np.exp(-(r-m)**2/(2*s**2))
    iNaNs = np.isnan(P2D)
    P2D[iNaNs] = num_min
    P2D[P2D < num_min] = num_min
    return P2D   
    
def P2D_gen(m, s, N):
    "Generate N random variable of mu from P2D distribution"
    r = np.linspace(max(num_min, m-10*s), m+10*s, num=100000)
    if m < 10*s:
        pdf = P2D(m, s, r)
    else:
        pdf = P2D_approx(m, s, r)
    pdf = pdf/sum(pdf)
    return np.random.choice(r, size=N, p=pdf).tolist()
    

def sigma_gen(m, s, N):
    "Generate N random variable of sigma from gamma distribution"
    shape = (m/s)**2.0
    scale = (s**2.0)/m
    return np.random.gamma(shape, scale, size=N).tolist()
                
# Class Mol 
class Mol(object):
    def __init__(self, m, s, j):
        self.group_num = j  
        self.m = m
        self.s = s   
        self.x_true = m
        self.y_true = 0        
        self.x_loc = np.random.normal(self.x_true, self.s)
        self.y_loc = np.random.normal(self.y_true, self.s)        
        self.r_loc = (self.x_loc**2.0 + self.y_loc**2.0)**0.5

                                                
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
                'mols' : [],
                'm' : [],
                's' : [],                
                'x_loc' : [],
                'y_loc' : [],
                'r_loc' : []               
            }
                  
    def generate_data(self):
        self.m_max, self.m_min = 0, 1000
        self.s_max, self.s_min = 0, 1000
        self.x_max, self.x_min = 0, 1000
        self.y_max, self.y_min = 0, 1000
        self.r_max, self.r_min = 0, 1000       
        
        for i in range(self.N_group):
            g = self.group[i]      
            g['m'] = P2D_gen(g['m_m'], g['m_s'], g['N'])    
            g['s'] = sigma_gen(g['s_m'], g['s_s'], g['N']) 

            # Generate N molecules with parameters 
            for j in range(g['N']):
                mol = Mol(g['m'][j], g['s'][j], i)
                g['mols'].append(mol)
                g['x_loc'].append(mol.x_loc)
                g['y_loc'].append(mol.y_loc)
                g['r_loc'].append(mol.r_loc)
            self.m_max = max(self.m_max, max(g['m']))
            self.m_min = min(self.m_min, min(g['m']))
            self.s_max = max(self.s_max, max(g['s']))
            self.s_min = min(self.s_min, min(g['s']))
            self.x_max = max(self.x_max, max(g['x_loc']))
            self.x_min = min(self.x_min, min(g['x_loc']))
            self.y_max = max(self.y_max, max(g['y_loc']))
            self.y_min = min(self.y_min, min(g['y_loc']))            
            self.r_max = max(self.r_max, max(g['r_loc']))
            self.r_min = min(self.r_min, min(g['r_loc']))
            
    def analyze_data(self):
        pass
                       
              
    def plot_data(self):
        plt.close('all')
        self.fig1 = plt.figure(1)
        
        # Distribution of mu
        sp = self.fig1.add_subplot(231) 
        m = []
        bins = np.linspace(self.m_min, self.m_max, num_bin)
        for i in range(self.N_group):   
            g = self.group[i]
            m += g['m']
            sp.hist(g['m'], bins, normed=False, color=g['c'], histtype='step', linewidth=1)
        sp.hist(m, bins, normed=False, color='k', histtype='step', linewidth=1)
        sp.set_title('Mu')
                      
        # Distribution of sigma
        sp = self.fig1.add_subplot(232) 
        s = []
        bins = np.linspace(self.s_min, self.s_max, num_bin)
        for i in range(self.N_group):   
            g = self.group[i]
            s += g['s']
            sp.hist(g['s'], bins, normed=False, color=g['c'], histtype='step', linewidth=1)
        sp.hist(s, bins, normed=False, color='k', histtype='step', linewidth=1)
        sp.set_title('Sigma')

        # Scatter plot of the groups
        sp = self.fig1.add_subplot(233) 
        for i in range(self.N_group):   
            g = self.group[i]
            c = g['c']
            sp.plot(g['x_loc'], g['y_loc'], c+'.', markersize=1, alpha=0.5)
        sp.axvline(x=0, color='k', linewidth=0.5)
        sp.axhline(y=0, color='k', linewidth=0.5)
        sp.set_aspect('equal')

        # Distribution in x
        sp = self.fig1.add_subplot(234)
        x_loc = []
        bins = np.linspace(self.x_min, self.x_max, num_bin)
        for i in range(self.N_group):   
            g = self.group[i]
            x_loc += g['x_loc']
            sp.hist(g['x_loc'], bins, normed=False, color=g['c'], histtype='step', linewidth=1)
        sp.hist(x_loc, bins, normed=False, color='k', histtype='step', linewidth=1)
        sp.set_title('X localization')        

        # Distribution in y
        sp = self.fig1.add_subplot(235)
        y_loc = []
        bins = np.linspace(self.y_min, self.y_max, num_bin)
        for i in range(self.N_group):   
            g = self.group[i]
            y_loc += g['y_loc']
            sp.hist(g['y_loc'], bins, normed=False, color=g['c'], histtype='step', linewidth=1)
        sp.hist(y_loc, bins, normed=False, color='k', histtype='step', linewidth=1)
        sp.set_title('Y localization')   


        sp = self.fig1.add_subplot(236)
        r = []
        bins = np.linspace(self.r_min, self.r_max, num_bin)
        for i in range(self.N_group):   
            g = self.group[i]
            r += g['r_loc']
            sp.hist(g['r_loc'], bins, normed=False, color=g['c'], histtype='step', linewidth=1)
        sp.hist(r, bins, normed=False, color='k', histtype='step', linewidth=1)
        sp.set_title('Euclidean distance')
                    

# Parameters [N, m_m, m_s, s_m, s_s, c]
# N: Number of particle
# m_m: mu, mean
# m_s: mu, std
# s_m: sigma, mean
# s_m: sigma, std
# c: color

g1 = {'N':100000, 'm_m':10, 'm_s':10, 's_m':1, 's_s':0.1, 'c':'b'}
g2 = {'N':100000, 'm_m':15, 'm_s':5, 's_m':1, 's_s':0.1, 'c':'r'}
groups = [g1, g2]

sample = Sample(groups)
t1 = time.clock()
sample.generate_data()
t2 = time.clock()
sample.plot_data()
t3 = time.clock()
sample.analyze_data()
t4 = time.clock()
plt.show()

print(t2-t1)
print(t3-t2)
print(t4-t3)
            
    