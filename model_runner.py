"""Solver adapter; no shared output files or extrapolated points in profiles."""
import warnings
import numpy as np
import pandas as pd
from fast_solver import starmodel
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
    core={n:table[-1][n].item() for n in table.dtype.names}
    # Leading-order central volume integral using the innermost shell's local
    # density and energy generation, not rho_core=M_shell/volume (which would
    # force zero residual by construction).
    candidates=df[(df.r>0)&np.isfinite(df[['r','rho','epsilon','M','L']]).all(axis=1)]
    if len(candidates):
        shell=candidates.iloc[0]
        core_volume=4*np.pi/3*shell.r**3
        core['M']=float(shell.M-core_volume*shell.rho)
        core['L']=float(shell.L-core_volume*shell.rho*shell.epsilon)
        core['Qm']=1-core['M']/(mass*1.989e33)
    return dict(parameters=(mass,luminosity,teff,x,z), flag=int(flag), error=int(error),
                profile=df, core=core,
                radius=radius, warnings=list(dict.fromkeys(str(w.message) for w in caught)))

EXPLANATIONS = {
    -1: 'The solver reached its shell limit before establishing a consistent core. This may be a numerical limitation rather than a simple mismatch in your trial parameters.',
    0: 'The extrapolated core satisfies the solver’s density, energy generation, and temperature checks. Inspect the profiles and remaining mass and luminosity as well.',
    1: 'The density inferred from the remaining mass and core volume is below the last shell’s density or above the solver’s allowed upper limit. The inferred core does not connect smoothly to the integrated interior.',
    2: 'The core energy generation per unit mass inferred from remaining luminosity divided by remaining mass is lower than the last shell’s value. The core cannot connect consistently to the luminosity profile.',
    3: 'The extrapolated central temperature is lower than the last integrated shell’s temperature. The temperature should rise toward the center in this model.',
    4: 'Enclosed mass became negative before the center was reached. Integrating the trial density profile inward consumed more mass than the star was assigned.',
    5: 'Luminosity became negative before the center was reached. Integrating the trial energy generation inward consumed more luminosity than the assumed surface luminosity.',
    6: 'The integration reached or crossed zero radius with mass or luminosity still remaining. The trial surface conditions do not produce a consistent center.',
}

def diagnostics(result):
    """Use the last finite shell at positive radius, never a core extrapolation."""
    p = result['profile']
    valid = (p.r > 0) & np.isfinite(p[['r_fraction','m_fraction','l_fraction']]).all(axis=1)
    if not valid.any():
        return {'r/R':np.nan, 'M/M★':np.nan, 'L/L★':np.nan}
    row=p.loc[valid].iloc[0]
    return {'r/R':float(row.r_fraction), 'M/M★':float(row.m_fraction), 'L/L★':float(row.l_fraction)}

def explanation(result):
    if result['error']:
        return 'The numerical integration encountered an equation-of-state or intermediate integration error. The integration stopped before establishing a valid core; the condition flag alone does not establish the cause.'
    if result['flag']==1:
        shells=result['profile']
        shells=shells[(shells.r>0)&np.isfinite(shells.rho)&(shells.rho>0)]
        if len(shells)>=2:
            inner=float(shells.iloc[0].rho)
            upper=10*inner*inner/float(shells.iloc[1].rho)
            core=result['core']['rho']
            direction='below the last shell density' if core<inner else 'above the allowed upper limit'
            return (f'The remaining mass and core volume imply a mean core density of {core:.5g} g/cm³, '
                    f'{direction}. The allowed range is {inner:.5g}–{upper:.5g} g/cm³. '
                    'Small mass and luminosity residuals alone do not ensure a consistent density profile.')
    return EXPLANATIONS.get(result['flag'], 'The solver returned an unrecognized condition flag.')
