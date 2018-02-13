from __future__ import division, print_function, absolute_import
import numpy as np
import matplotlib.pyplot as plt
import scipy.special as sp

pi = 3.141592

def P2D1(m, s, r): 
    P2D = (r/(2*pi*s*m))**0.5 * np.exp(-(r-m)**2/(2*s**2))
#    iNaNs = np.isnan(P2D)
#    P2D[iNaNs] = 1e-100
    return P2D
    
def P2D2(m, s, r): 
    P2D = (r/s**2)*np.exp(-(m**2 + r**2)/(2*s**2))*sp.i0(r*m/s**2)  
#    iNaNs = np.isnan(P2D)
#    P2D[iNaNs] = 1e-100
    return P2D
    

m = 10
s = 1

r = np.linspace(max(1e-100, m-10*s), m+10*s, num=100000)
    
pdf1 = P2D1(m, s, r)
pdf1 = pdf1/sum(pdf1)
    
pdf2 = P2D2(m, s, r)
pdf2 = pdf2/sum(pdf2)    
    
pdf12 = pdf1 - pdf2

plt.close()
plt.figure()
plt.plot(r, pdf1, 'b', r, pdf2, 'r')
plt.show()


