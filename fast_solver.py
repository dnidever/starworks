"""Numba-backed STATSTAR with the same structured result as the reference solver."""
import numpy as np
from fast_kernel import integrate
DTYPE=np.dtype([('index',int),('r',float),('Qm',float),('L',float),('T',float),('P',float),
                ('rho',float),('M',float),('kappa',float),('epsilon',float),('zone','U1'),('dlnPdlnT',float)])
def starmodel(mass,luminosity,teff,x,z,verbose=False):
    flag,error,stop,r,p,m,l,t,rho,kappa,eps,gradient,rhoc,epsc,pc,tc=integrate(mass,luminosity,teff,x,z)
    n=stop+1
    tab=np.zeros(n+1,dtype=DTYPE)
    tab['index']=np.arange(1,n+2)
    for name,values in [('r',r),('P',p),('M',m),('L',l),('T',t),('rho',rho),('kappa',kappa),('epsilon',eps)]:
        tab[name][:n]=values[:n]
    tab['Qm'][:n]=1-m[:n]/(mass*1.989e33)
    tab['zone'][:n]=np.where(gradient[:n]<(5/3)/((5/3)-1),'c','r')
    tab['dlnPdlnT'][:n]=np.clip(gradient[:n],-99.9,99.9)
    tab[-1]=tab[n-1]
    tab['index'][-1]=n+1
    for name,value in [('r',0),('rho',rhoc),('epsilon',epsc),('P',pc),('T',tc)]:tab[name][-1]=value
    return flag,error,stop,tab
