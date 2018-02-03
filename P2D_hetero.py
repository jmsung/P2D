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
import random 
from scipy.stats import gamma
from scipy.stats import chisqprob

# P2D(r|m,s) = (r/s^2)*exp(-(m^2+r^2)/(2*s^2))*I0(r*m/s^2) Eq. (4)
# I0 = Modified Bessel function of order 0.

def P2D(m, s, r):   
    return (r/s**2)*np.exp(-(m**2 + r**2)/(2*s**2))*sp.i0(r*m/s**2)

# LogLikelihood 
def LL(param, s, r):
    m = np.abs(param)      
    return np.sum(np.log10(P2D(m, s, r)))

def MLE(m, s, r): # P2D MLE with fixed mean sigma
    fun = lambda *args: -LL(*args)
    p0 = [m]
    result = minimize(fun, p0, method='SLSQP', args=(s, r)) 
    return result
    
    
# Class Mol 
class Mol(object):
    def __init__(self):
        pass       
        
    
# Class Sample with multiple molecules    
class Sample(object):
    def __init__(self, N, m_m, m_s, s_m, s_s):
        self.N_state = len(N)
        self.N = []
        self.m_m = []
        self.m_s = []
        self.s_m = []
        self.s_s = []
        
        for i in range(self.N_state):
            self.N.append(N[i])
            self.m_m.append(m_m[i])
            self.m_s.append(m_s[i])
            self.s_m.append(s_m[i])
            self.s_s.append(s_s[i])            
          
        
    def generate(self):
        pass
        
    def analyze(self):
        pass

# Parameters
N = [1000]
m_m = [10]
m_s = [1]
s_m = [10]
s_s = [3]

sample = Sample(N, m_m, m_s, s_m, s_s)
sample.generate()
sample.analyze()
    
            
    