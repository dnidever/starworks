import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st
from model_runner import run_model, STATUS, diagnostics, explanation
st.set_page_config(page_title='StarWorks',page_icon='☀️',layout='wide')
st.title('StarWorks')
st.caption('Explore homogeneous main-sequence models with STATSTAR. Adjust the surface conditions, integrate inward, and inspect the interior.')
@st.cache_data(max_entries=200,show_spinner=False)
def calculate(*pars):
    return run_model(*pars)
with st.sidebar:
    st.header('Model parameters')
    with st.form('parameters'):
        mass=st.number_input('Mass (M☉)',min_value=0.1,max_value=100.0,value=1.0,step=0.1,format='%.4f')
        lum=st.number_input('Luminosity (L☉)',min_value=0.00001,max_value=1000000.0,value=0.86071,step=0.001,format='%.5f')
        teff=st.number_input('Effective temperature (K)',min_value=1000.0,max_value=100000.0,value=5500.2,step=1.0,format='%.1f')
        x=st.number_input('Hydrogen mass fraction X',min_value=0.001,max_value=0.999,value=0.70,step=0.01,format='%.3f')
        z=st.number_input('Metal mass fraction Z',min_value=0.000001,max_value=0.5,value=0.008,step=0.001,format='%.6f')
        submitted=st.form_submit_button('Run model',type='primary')
    st.caption('Helium fraction Y = 1 − X − Z. Luminosity and temperature are trial boundary conditions, not predictions.')
    st.caption('Initial inputs reproduce the 1 M☉ trial from your notebook.')
if submitted:
    try:
        with st.spinner('Integrating stellar structure…'):
            st.session_state.current=calculate(mass,lum,teff,x,z)
    except Exception as exc:
        st.error(f'Model failed: {exc}')
from grid_search import grid_search_ui
grid_search_ui(calculate,(mass,x,z),st.session_state.get('current'))
if 'current' not in st.session_state:
    st.info('Choose parameters and click Run model to explore the stellar interior.')
    st.stop()
r=st.session_state.current
m,l,t,h,met=r['parameters']
st.caption(f'Displayed model: M={m:g} M☉ · L={l:g} L☉ · Teff={t:g} K · X={h:g} · Y={1-h-met:g} · Z={met:g}')
msg=STATUS.get(r['flag'],f"Unknown status {r['flag']}")
if r['flag']==0 and r['error']==0:
    st.success(msg+'; inspect the profiles and residuals before accepting the model.')
else:
    st.warning(f"{msg}. Integration error code: {r['error']}. Profiles show a trial model, not an accepted solution.")
st.markdown('**What happened in the core?**')
st.write(explanation(r))
st.caption('The residuals below describe the innermost finite shell at positive radius. They are not the mass or luminosity of a point at the center.')
df=r['profile']; inner=diagnostics(r)
cols=st.columns(4)
for col,label,value in zip(cols,['Radius (R☉)','Innermost r/R','Remaining M/M★','Remaining L/L★'],[r['radius']/6.9599e10,inner['r/R'],inner['M/M★'],inner['L/L★']]):
    col.metric(label,f'{value:.4g}')
st.subheader('Which direction should I try?')
st.write('Test small changes in luminosity and effective temperature while keeping mass and composition fixed. These tests use the displayed model, even if you have edited the sidebar inputs.')
step=st.number_input('Adjustment size (%)',min_value=0.01,max_value=10.0,value=1.0,step=0.1)
if st.button('Test adjustment directions'):
    trials=[('Baseline',l,t),('Increase luminosity',l*(1+step/100),t),('Decrease luminosity',l*(1-step/100),t),('Increase temperature',l,t*(1+step/100)),('Decrease temperature',l,t*(1-step/100))]
    rows=[]
    with st.spinner('Testing nearby surface conditions…'):
        for name,trial_l,trial_t in trials:
            row={'Trial':name,'L (L☉)':trial_l,'Teff (K)':trial_t}
            try:
                trial=calculate(m,trial_l,trial_t,h,met)
                d=diagnostics(trial)
                row.update(d)
                row['Status']=STATUS.get(trial['flag'],'Unknown') if not trial['error'] else 'Numerical integration error'
                row['Integration error']=trial['error']
                row['Δ|M/M★|']=abs(d['M/M★'])-abs(inner['M/M★'])
                row['Δ|L/L★|']=abs(d['L/L★'])-abs(inner['L/L★'])
            except Exception as exc:
                row['Status']=f'Could not calculate: {exc}'
            rows.append(row)
    st.session_state.direction_tests={'parameters':r['parameters'],'step':step,'rows':rows}
tests=st.session_state.get('direction_tests')
if tests and tests['parameters']==r['parameters'] and tests['step']==step:
    st.dataframe(pd.DataFrame(tests['rows']),hide_index=True,width='stretch')
    st.caption('Negative Δ means a smaller absolute residual. Compare r/R first: trials that stop at different radii are not directly comparable. Even both residuals shrinking does not guarantee convergence.')
    st.info('If a trial passes the core checks without an integration error, inspect it next. Otherwise, look for a change that improves the residuals at a similar stopping radius. Try that direction with a smaller step, change one parameter at a time, and rerun. If mass and luminosity respond in opposite ways, both surface parameters may need tuning.')
    st.write('Physical clue: negative luminosity means the trial interior used up its luminosity too early; increasing the assumed luminosity is one experiment to test. There is no universal temperature adjustment rule because both inputs change the entire interior.')

a,b=st.columns([3,1])
with a:
    axis=st.radio('Horizontal axis',['Fractional radius','Enclosed mass fraction'],horizontal=True)
with b:
    log=st.checkbox('Logarithmic positive profiles',value=True)
log_x=st.toggle('Logarithmic x-axis',value=False)
if log_x:
    st.caption('The logarithmic x-axis shows positive coordinates only; zero and negative coordinates are omitted.')
coord='r_fraction' if axis=='Fractional radius' else 'm_fraction'
x_range=None
if st.toggle('Limit x-range',value=False):
    cmin,cmax=st.columns(2)
    xmin=cmin.number_input('Minimum x',value=0.001 if log_x else 0.0,format='%.6f')
    xmax=cmax.number_input('Maximum x',value=0.3,format='%.6f')
    if not np.isfinite([xmin,xmax]).all() or xmin>=xmax or (log_x and xmin<=0):
        st.error('Use minimum < maximum, with positive bounds for a logarithmic x-axis.')
    else:
        x_range=(xmin,xmax)

show_core=st.toggle('Show extrapolated core point',value=True)
if show_core:
    st.caption('Open diamonds show the extrapolated core. Core mass and luminosity are extrapolated residuals using the last shell’s density and energy generation held constant over the remaining core volume; opacity and gradient are copied from the last shell. The zero-radius point cannot appear on a logarithmic radius axis.')
reference=st.session_state.get('reference')
c1,c2=st.columns(2)
if c1.button('Keep current model as comparison'):
    st.session_state.reference=r
    reference=r
if c2.button('Clear comparison'):
    st.session_state.pop('reference',None)
    reference=None
from plots import make_profiles, invalid_shells
st.plotly_chart(make_profiles(r,reference,coord,axis,log,log_x=log_x,x_range=x_range,show_core=show_core),width='stretch')
st.caption('Temperature and density are divided by their own positive maxima; hover to see actual values. Comparison models use their own maxima. Extrapolated core points appear only when enabled.')
if invalid_shells(df).any():
    st.warning('Red shading and solid red lines mark shells with negative radius, mass, luminosity, opacity or energy generation; nonpositive temperature, pressure or density; or nonfinite values. These are invalid computed values, not a physical stellar region. Nonpositive values are omitted on logarithmic axes. On the mass axis, only lines are used because the coordinate can reverse in failed models.')
if r['flag']!=0 or r['error']:
    st.caption('Red inner shading on the radius axis marks the unresolved interior from the last valid shell to the center, not a computed physical region. On a logarithmic axis it extends to the visible lower edge. The dashed red line marks the innermost finite positive-radius shell of the failed integration.')
if reference and (reference['flag']!=0 or reference['error']):
    st.caption('The comparison is also a failed trial. Red diagnostic markers apply to the current model only.')
st.caption('The starting surface shells are assumed radiative; transport labels come from the solver.')
with st.expander('Advanced plots: pressure, opacity and temperature gradient'):
    st.plotly_chart(make_profiles(r,reference,coord,axis,log,advanced=True,log_x=log_x,x_range=x_range,show_core=show_core),width='stretch')
with st.expander('Extrapolated core and numerical diagnostics'):
    st.write('These values extrapolate from the last integrated shell. Central mass and luminosity below are extrapolated residuals, not physical point values. They use M₀ ≈ Mᵢ − (4π/3)ρᵢrᵢ³ and L₀ ≈ Lᵢ − (4π/3)ρᵢεᵢrᵢ³. This is a leading-order approximation, not another integration.')
    st.json({k:r['core'][k] for k in ['T','rho','P','epsilon','M','L']})
    st.write({'condition_flag':r['flag'],'integration_error':r['error'],'integrated_shells':len(df)})
    if r['warnings']: st.write(r['warnings'])
with st.expander('Model table and downloads'):
    display_table=df.rename(columns={'r_fraction':'Fractional radius (r/R)'})
    st.dataframe(display_table,width='stretch')
    st.download_button('Download integrated shells (CSV)',df.to_csv(index=False),'starmodel_shells.csv','text/csv')
with st.expander('How to use this in class'):
    st.markdown('Keep a model as a comparison, then change one parameter. Examine where the luminosity rises, where energy transport is convective, and how concentrated the mass is. Adjust luminosity and effective temperature to reduce the remaining central mass and luminosity while meeting the core checks. This simplified solver assumes homogeneous composition and is not an evolutionary model.')
