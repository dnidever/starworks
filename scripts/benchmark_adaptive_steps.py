"""Isolated step-policy experiment. Does not modify the production solver.
Run from the repository root: python scripts/benchmark_adaptive_steps.py
"""
import importlib.util
import tempfile
import time
from pathlib import Path
import sys
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from fast_kernel import integrate

POLICY='''        # Reuse the four derivatives; no extra equation evaluations.
        step = Rs / 200.0
        scales = (abs(P[im1]), max(abs(M_r[im1]), 0.0001*Ms),
                  max(abs(L_r[im1]), 0.001*Ls), abs(T[im1]))
        for component in range(4):
            if abs(dfdr[component]) > 0.0:
                step = min(step, FRACTION * scales[component] / abs(dfdr[component]))
        # Grow at most 20% per shell. A floor prevents vanishing steps.
        step = min(step, 1.2*abs(deltar), 0.5*r[im1])
        deltar = -max(step, Rs/25000.0)
'''
TRANSITION='''        if idrflg == 0 and M_r[i] < 0.99 * Ms:
            deltar = -1.0 * Rs / 1000.0
            idrflg = 1
        if idrflg == 1 and abs(deltar) >= 0.5 * r[i]:
            deltar = -1.0 * Rs / 5000.0
            idrflg = 2
'''

def load_variant(directory,name,source):
    path=Path(directory)/f'{name}.py'
    path.write_text(source.replace('cache=True','cache=False'))
    spec=importlib.util.spec_from_file_location(name,path)
    module=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.integrate

def main():
    source=(ROOT/'fast_kernel.py').read_text()
    assert TRANSITION in source
    anchor='        f_i, ierr = runge(f_im1, dfdr, r[im1], deltar, irc, X, Z, XCNO, mu, i)'
    pars=[(1.,float(l),float(t),.7,.008) for l in np.linspace(.862,.868,7) for t in np.linspace(5503.5,5523.5,7)]
    solar=(1.,.8652,5513.5,.7,.008)
    pars.append(solar)
    with tempfile.TemporaryDirectory() as directory:
        fine=source.replace('5000','25000').replace('1000.0','5000.0')
        variants=[('current',integrate),('fine_fixed',load_variant(directory,'fine_fixed',fine))]
        for fraction in [.02,.05,.1]:
            adaptive=source.replace(TRANSITION,'').replace(anchor,POLICY.replace('FRACTION',str(fraction))+anchor)
            variants.append((f'adaptive_{fraction}',load_variant(directory,'adaptive_'+str(int(fraction*100)),adaptive)))
        results={}
        for name,fn in variants:
            fn(*solar) # compile outside timings
            start=time.perf_counter()
            rows=[fn(*p) for p in pars]
            elapsed=time.perf_counter()-start
            results[name]=rows
            trial=fn(*solar)
            start=time.perf_counter()
            for _ in range(200):fn(*solar)
            solar_ms=(time.perf_counter()-start)*5
            accepted=sum(r[0]==0 and r[1]==0 for r in rows)
            print(f'{name}: solar flag/error={trial[:2]}, shells={trial[2]}, solar_ms={solar_ms:.4f}, grid_ms={elapsed*1000:.2f}, passing={accepted}/{len(pars)}',flush=True)
        reference=results['fine_fixed']
        for name,rows in results.items():
            agreement=sum(r[:2]==ref[:2] for r,ref in zip(rows,reference))
            accepted_agreement=sum((r[0]==0 and r[1]==0)==(ref[0]==0 and ref[1]==0) for r,ref in zip(rows,reference))
            # Compare T and rho at common radii outside the extrapolated core.
            errors=[]
            query=np.linspace(.03,.95,150)
            for row,ref in zip(rows,reference):
                for column in [7,8]:
                    profiles=[]
                    for model in [row,ref]:
                        radius=model[3][:model[2]+1]/model[3][0]
                        values=model[column][:model[2]+1]
                        valid=(radius>0)&np.isfinite(values)&(values>0)
                        order=np.argsort(radius[valid])
                        profiles.append((radius[valid][order],values[valid][order]))
                    (ra,va),(rb,vb)=profiles
                    if len(ra)<2 or len(rb)<2:continue
                    q=query[(query>=max(ra[0],rb[0]))&(query<=min(ra[-1],rb[-1]))]
                    if len(q):
                        actual=np.interp(q,ra,va);expected=np.interp(q,rb,vb)
                        errors.extend(((actual-expected)/expected)**2)
            print(f'{name}: T/rho relative RMS vs finer reference={np.sqrt(np.mean(errors))*100:.4f}%',flush=True)
            print(f'{name}: finer-reference status agreement={agreement}/{len(pars)}, pass/fail agreement={accepted_agreement}/{len(pars)}',flush=True)

if __name__=='__main__':main()
