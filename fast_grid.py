"""Compile the grid loop too; produce only flags and last-shell diagnostics."""
import numpy as np
from numba import njit
from fast_kernel import integrate
# Kernel revision: main radial step R/1000, core R/5000, shell limit 5000.
@njit(cache=False,error_model='numpy')
def grid_summary(mass,x,z,luminosities,temperatures,include_core=False):
    output=np.empty((len(luminosities)*len(temperatures),11 if include_core else 7))
    j=0
    for luminosity in luminosities:
        for temperature in temperatures:
            result=integrate(mass,luminosity,temperature,x,z)
            flag,error,stop=result[0],result[1],result[2]
            r,m,l=result[3],result[5],result[6]
            best=-1
            for k in range(stop+1):
                if r[k]>0 and np.isfinite(r[k]) and np.isfinite(m[k]) and np.isfinite(l[k]):
                    if best<0 or r[k]<r[best]:best=k
            output[j,0]=luminosity;output[j,1]=temperature
            output[j,2]=flag;output[j,3]=error
            output[j,4:7]=np.nan
            if best>=0:
                output[j,4]=r[best]/r[0]
                output[j,5]=m[best]/(mass*1.989e33)
                output[j,6]=l[best]/(luminosity*3.826e33)
            if include_core:
                output[j,7:11]=np.nan
                if error==0 and best>0 and result[8][best]>0 and result[8][best-1]>0 and m[best]>0 and result[7][best]>0:
                    rhoc=result[12]
                    rhomin=result[8][best]
                    rhomax=10*rhomin*rhomin/result[8][best-1]
                    output[j,7]=rhoc/rhomin
                    output[j,8]=rhoc/rhomax
                    if result[10][best]>0:output[j,9]=result[13]/result[10][best]
                    output[j,10]=result[15]/result[7][best]
            j+=1
    return output
