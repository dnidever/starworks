"""Solver adapter; no shared output files or extrapolated points in profiles."""
import warnings
import numpy as np
import pandas as pd
from starmodel import starmodel
STATUS = {-1:'Maximum shell count reached',0:'Passed solver core checks',1:'Core density mismatch',2:'Core energy generation mismatch',3:'Extrapolated core temperature too low',4:'Negative enclosed mass',5:'Negative luminosity',6:'Center reached with remaining mass or luminosity'}
def run_model(mass, luminosity, teff, x, z):
    if min(mass, luminosity, teff) <= 0 or not (0 < x < 1 and 0 < z < 1 and x+z < 1):
        raise ValueError('Use positive mass, luminosity, temperature, X and Z, with X + Z < 1.')
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        flag, error, stop, table = starmodel(mass,luminosity,teff,x,z,verbose=False)
    df = pd.DataFrame(table[:-1]).sort_values('r').reset_index(drop=True)
    radius = float(df.r.max())
    df['r_fraction'] = df.r / radius
    df['m_fraction'] = df.M / (mass*1.989e33)
    df['l_fraction'] = df.L / (luminosity*3.826e33)
    return dict(parameters=(mass,luminosity,teff,x,z), flag=int(flag), error=int(error),
                profile=df, core={n:table[-1][n].item() for n in table.dtype.names},
                radius=radius, warnings=list(dict.fromkeys(str(w.message) for w in caught)))
