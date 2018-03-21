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
from scipy.stats import chisqprob
import time

pi = 3.141592
num_min = 1e-100
num_bin = 50
color = ['b', 'r', 'g', 'c', 'm', 'y', 'b', 'r', 'g', 'c', 'm', 'y', 'b', 'g', 'r', 'c', 'm', 'y']

  
def MLE(m, s, r): # P2D with fixed mean sigma
    fun = lambda *args: LL(*args)
    p0 = [m]
    result = minimize(fun, p0, method='SLSQP', args=(s, r)) 
    LL = result["fun"]
    m = result["x"]
    return LL, m
    
def LL(m, s, r):
    "Negative LogLikelihood per molecule"   
    return -np.sum(np.log10(P2D(m, s, r)))/len(r)

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
 
def RMSD(x, y):
    x = np.array(x)
    y = np.array(y)
    return (np.mean((x-y)**2))**0.5                
                                              
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
                  
    def generate(self): 
        self.g_simul = []
        self.mols_simul = []      
        self.m_simul = []
        self.s_simul = []
        self.x_simul = []
        self.y_simul = []
        self.r_simul = []
        
        for i in range(self.N_group):
            g = self.group[i]      
            g['m'] = P2D_generate(g['m_m'], g['m_s'], g['N'])    
            g['s'] = sigma_generate(g['s_m'], g['s_s'], g['N']) 

            # Generate N molecules with parameters 
            for j in range(g['N']):
                mol = Mol(g['m'][j], g['s'][j], i)
                self.mols_simul.append(mol)
                g['x'] += [mol.x]
                g['y'] += [mol.y]
                g['r'] += [mol.r]    
                  
            self.g_simul += [i]         
            self.m_simul += g['m']
            self.s_simul += g['s']
            self.x_simul += g['x']
            self.y_simul += g['y']
            self.r_simul += g['r']  
                    
        self.n_simul = len(self.m_simul)                           
                                                                                                                                               
    def plot_simul(self):
        plt.close('all')
        self.fig1 = plt.figure(1)
        
        m_simul = self.m_simul
        m_simul_guess = self.m_simul_guess
        s_simul = self.s_simul
        r_simul = self.r_simul       
        LL_simul = self.LL_simul
        dm_simul = self.dm_simul
                        
        r_min = min(r_simul)
        r_max = min(r_simul)
        s_min = min(s_simul)
        s_max = max(s_simul)
        m_min = min(min(m_simul), min(m_simul_guess))
        m_max = max(min(m_simul), max(m_simul_guess))
                                                                                                                  
        # Estimation of Mu_simul
        sp = self.fig1.add_subplot(251)
        bins = np.linspace(m_min, m_max, num_bin)
        m = ''
        for i in range(self.N_group):   
            g = self.group[i]
            sp.hist(g['m'], bins, normed=False, color=g['c'], histtype='step', linewidth=1)
            m += '(%.1f +/- %.1f) ' %(g['m_m'], g['m_s'])            
        sp.hist(m_simul, bins, normed=False, color='k', histtype='step', linewidth=1)
        sp.hist(m_simul_guess, bins, normed=False, color='r', histtype='step', linewidth=1)
        title = 'M (simul) = %s' % (m)
        sp.set_title(title)                         
                                                                           
        # Distribution of sigma_simul
        sp = self.fig1.add_subplot(252) 
        bins = np.linspace(s_min, s_max, num_bin)
        s = ''
        for i in range(self.N_group):   
            g = self.group[i]
            sp.hist(g['s'], bins, normed=False, color=g['c'], histtype='step', linewidth=1)
            s += '(%.1f +/- %.1f) ' %(g['s_m'], g['s_s'])   
        sp.hist(s_simul, bins, normed=False, color='k', histtype='step', linewidth=1)
        title = 'S (simul) = %s' % (s)
        sp.set_title(title)

        # Distributio of r_simul
        sp = self.fig1.add_subplot(253)
        bins = np.linspace(r_min, r_max, num_bin)
        N = ''
        for i in range(self.N_group):   
            g = self.group[i]
            sp.hist(g['r'], bins, normed=False, color=g['c'], histtype='step', linewidth=1)
            N += '(%d) ' %(g['N']) 
        sp.hist(r_simul, bins, normed=False, color='k', histtype='step', linewidth=1)
        title = 'R (simul), N = %s ' % (N)
        sp.set_title(title)                                          
       
        # LL_simul change over iteration
        sp = self.fig1.add_subplot(254)
        sp.plot(LL_simul, 'k')
        sp.set_title('-LL (simul)')
        
        # dm_simul change over iteration
        sp = self.fig1.add_subplot(255)
        sp.plot(dm_simul, 'k')
        sp.set_title('RMSD of m (simul)')        

    def plot_data(self):
        plt.close('all')
        self.fig2 = plt.figure(2)
        
        m_simul = self.m_simul
        m_simul_guess = self.m_simul_guess
        s_simul = self.s_simul
        r_simul = self.r_simul       
        LL_simul = self.LL_simul
        dm_simul = self.dm_simul
                 
        s_data = self.s_data
        r_data = self.r_data
        m_data_guess = self.m_data_guess
        LL_data = self.LL_data
        
        r_min = min(min(r_simul), min(r_data))
        r_max = max(min(r_simul), max(r_data))
        s_min = min(min(s_simul), min(s_data))
        s_max = max(min(s_simul), max(s_data))
        m_min = min(min(m_simul), min(m_simul_guess), min(m_data_guess))
        m_max = max(min(m_simul), max(m_simul_guess), max(m_data_guess))


        # Estimation of Mu_data
        sp = self.fig2.add_subplot(256)
        bins = np.linspace(m_min, m_max, num_bin)        
        sp.hist(m_data_guess, bins, normed=False, color='r', histtype='step', linewidth=1)
        title = 'M (data)' 
        sp.set_title(title)   
                                                                                                 
        # Distribution of sigma_data
        sp = self.fig2.add_subplot(257) 
        bins = np.linspace(s_min, s_max, num_bin) 
        sp.hist(s_data, bins, normed=False, color='k', histtype='step', linewidth=1)
        title = 'S (data) = %.1f +/- %.1f' % (np.mean(s_data), np.std(s_data))
        sp.set_title(title)
                                    
        # Distributio of r_data
        sp = self.fig2.add_subplot(258)
        bins = np.linspace(r_min, r_max, num_bin)
        sp.hist(r_data, bins, normed=False, color='k', histtype='step', linewidth=1)
        title = 'R (data), N = %s ' % (len(r_data))
        sp.set_title(title)      
        
        # LL_data change over iteration
        sp = self.fig2.add_subplot(259)
        sp.plot(LL_data, 'k')
        sp.set_title('-LL (data)')
                                                                                    
        self.fig2.tight_layout()
        self.fig2.subplots_adjust(wspace=0.2, hspace=0.2)    

                                                                                                                                                                                                     
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                               
    def analyze_simul(self):
        "Iteratively Gaussian sample new mu of each molecule and accept if LL increases."
        n_mols = self.n_simul
        n_iter = int(n_mols*1)
        r = np.array(self.r_simul)
        s = np.array(self.s_simul)
        m = np.array(self.m_simul)

        m_bin = np.linspace(min(r), max(r), num_bin)
        m_guess = np.random.choice(m_bin, size=n_mols)  
        step = np.mean(r)         
        LL_iter = [1e100] 
        dm = []
  
        start = time.clock()
        for i in range(n_iter):
            m_temp = m_guess.copy()
            m_temp[i%n_mols] = min(max(m_temp[i%n_mols] + np.random.randn(1)*step, 0), max(r))
            LL_temp = LL(m_temp, np.array(s), np.array(r))
            if LL_temp < LL_iter[-1]:
                LL_iter += [LL_temp]
                m_guess = m_temp
            else:
                LL_iter += [LL_iter[-1]]
                
            dm += [RMSD(m_guess, m)]
            
            done = (i+1)/n_iter*100
            if done%10 == 0:
                now = time.clock()
                spent = (now-start)/60 # time passed in min
                print('%d %%, %.1f min' %(done, (spent*(100-done)/done)))
                                                                                       
        self.LL_simul = np.array(LL_iter[1:])
        self.dm_simul = np.array(dm)
        self.m_simul_guess = m_guess
        
    def analyze_data(self): 
        "Iteratively Gaussian sample new mu of each molecule and accept if LL increases."
        n_mols = self.n_data
        n_iter = int(n_mols*1)
        r = np.array(self.r_data)
        s = np.array(self.s_data)

        m_bin = np.linspace(min(r), max(r), num_bin)
        m_guess = np.random.choice(m_bin, size=n_mols)  
        step = np.mean(r)         
        LL_iter = [1e100] 
  
        start = time.clock()
        for i in range(n_iter):
            m_temp = m_guess.copy()
            m_temp[i%n_mols] = min(max(m_temp[i%n_mols] + np.random.randn(1)*step, 0), max(r))
            LL_temp = LL(m_temp, np.array(s), np.array(r))
            if LL_temp < LL_iter[-1]:
                LL_iter += [LL_temp]
                m_guess = m_temp
            else:
                LL_iter += [LL_iter[-1]]
                            
            done = (i+1)/n_iter*100
            if done%10 == 0:
                now = time.clock()
                spent = (now-start)/60 # time passed in min
                print('%d %%, %.1f min' %(done, (spent*(100-done)/done)))
                                                                                         
        self.LL_data = np.array(LL_iter[1:])
        self.m_data_guess = m_guess
                      
                                                        
    def read_file(self, filename_r, filename_s):
        # Open file
        file_r = open(filename_r, "r") 
        file_s = open(filename_s, "r") 
        rl = []
        sl = []
    
        for line in file_r:
            rl.append(float(line))
        
        for line in file_s:
            sl.append(float(line))
    
        file_r.close()
        file_s.close()
    
        self.r_data = np.array(rl)
        self.s_data = np.array(sl)   
        self.n_data = len(self.r_data)   
 

# Simulation    
g1 = {'N':10000, 'm_m':10, 'm_s': 15, 's_m':3.4, 's_s':0.8, 'c':color[0]}
g2 = {'N':1000, 'm_m':20, 'm_s': 1, 's_m':3, 's_s':1, 'c':color[1]}
groups = [g1]

sample = Sample(groups)
sample.generate()
sample.read_file("CC1_r.txt", "CC1_s.txt")
sample.analyze_simul()
sample.analyze_data()
sample.plot_result()

plt.show()


"""    

def LL(m, s, r):
    "Negative LogLikelihood per molecule"  
    m = np.array(m)
    s = np.array(s)
    i1 = m > 10*s
    i2 = np.invert(i1)
    LL1 = np.sum(np.log10(P2D_approx(m[i1], s[i1], r[i1])))
    LL2 = np.sum(np.log10(P2D(m[i2], s[i2], r[i2])))
    return -(LL1 + LL2)/len(m)


    def analyze(self):
        "MLE with multiple conformations. Exchange conformation guess and evaluate LL"
        start = time.clock()
        n_mols = self.n_mols
        self.n_conf = 2
        n_iter = int(n_mols)*5
        m = np.array(self.m)
        s = np.array(self.s)
        r = np.array(self.r)

        # Initial guess of conf and m, based on r. Similar r are grouped together. 
        c_guess = np.zeros(n_mols)
        m_guess = np.zeros(n_mols)    
        r_bin = np.linspace(min(r), max(r), self.n_conf)
        if self.n_conf > 1:
            dr = (r_bin[1]-r_bin[0])/2
        else:
            dr = max(r) - min(r)

        for i in range(self.n_conf):
            j = (r > r_bin[i] - dr) & (r < r_bin[i] + dr)
            c_guess[j] = i
            m_guess[j] = np.mean(r[j])

        LL_init = 0
        mc_init = []
        mc_iter = []
        for i in range(self.n_conf): 
            k = i == c_guess # k = mols in i_th conformation 
            LL_k, m_k = MLE(np.mean(m_guess[k]), s[k], r[k])
            LL_init += LL_k
            mc_init.append(m_k)
            m_guess[k] = m_k

        LL_iter = [LL_init] # LL change over iteration
        dm_iter = [RMSD(m_guess, m)] # RMSD of (m and m_guess) over iteration        
        mc_iter.append(mc_init) # Mu of each conformation over iteration   

        if self.n_conf == 1:
            self.LL_iter2 = np.array(LL_iter)
            self.dm2 = np.array(dm_iter)
            self.mc2 = np.array(mc_iter)
            self.m_guess2 = m_guess
            self.c_guess2 = c_guess
            return True

        for i in range(n_iter): # i = current iteration
            j  = i%n_mols # j = mol to update conformation
            c1 = c_guess[j] # conformation (c1) of jth mol before change
            c2_possible = list(range(self.n_conf))
            c2_possible.remove(c1)
            c2 = np.random.choice(c2_possible, size=1) # New conformation (c2) for j-mol 

            c1_before = c1 == c_guess # Mols in c1 before changing
            c1_after = c1_before.copy()
            c1_after[j] = False # Mols in c1 after changing

            c2_before = c2 == c_guess # Mols in c2 before changing
            c2_after = c2_before.copy()
            c2_after[j] = True # Mols in c2 after changing 

            LL1_before, m1_before = MLE(np.mean(m[c1_before]), s[c1_before], r[c1_before])
            LL1_after,  m1_after  = MLE(np.mean(m[c1_after]),  s[c1_after],  r[c1_after])
            LL2_before, m2_before = MLE(np.mean(m[c2_before]), s[c2_before], r[c2_before])
            LL2_after,  m2_after  = MLE(np.mean(m[c2_after]),  s[c2_after],  r[c2_after])
                                
            # Accept update if LL increases with both conformational groups                   
            if (LL1_before + LL2_before > LL1_after + LL2_after):                    
                c_guess[j] = c2 # Update new conformation (c2) of j-mol 
                LL_new = 0
                mc_new = []
                for cf in range(self.n_conf): # cf = conformation of interest 
                    k = cf == c_guess # k = mols in cf conformation 
                    LL_k, m_k = MLE(np.mean(m_guess[k]), s[k], r[k])
                    LL_new += LL_k
                    mc_new.append(m_k)
                    m_guess[k] = m_k
                LL_iter += [LL_new]
                dm_iter += [RMSD(m_guess, m)]                
                mc_iter.append(mc_new)                 

            else:
                LL_iter += [LL_iter[-1]]       
                dm_iter += [RMSD(m_guess, m)]
                mc_iter.append(mc_iter[-1])
    
            done = (i+1)/n_iter*100
            if done%10 == 0:
                now = time.clock()
                spent = (now-start)/60 # time passed in min
                print('%d %%, %.1f min' %(done, (spent*(100-done)/done)))
               
        self.LL_iter = np.array(LL_iter)
        self.dm = np.array(dm_iter)
        self.mc = np.array(mc_iter)
        self.m_guess = m_guess
        self.c_guess = c_guess


"""

