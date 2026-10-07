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
for key,value in {'guess_mass':1.0,'guess_lum':0.86071,'guess_teff':5500.2,'guess_x':.70,'guess_z':.008}.items():
    st.session_state.setdefault(key,value)
pending=st.session_state.pop('pending_grid_guess',None)
if pending:
    for key,value in zip(['guess_mass','guess_lum','guess_teff','guess_x','guess_z'],pending):
        st.session_state[key]=float(value)
with st.sidebar:
    st.header('Model parameters')
    with st.form('parameters'):
        mass=st.number_input('Mass (M☉)',key='guess_mass',min_value=0.1,max_value=100.0,step=0.1,format='%.4f')
        lum=st.number_input('Luminosity (L☉)',key='guess_lum',min_value=0.000001,step=0.001,format='%.5f')
        teff=st.number_input('Effective temperature (K)',key='guess_teff',min_value=1.0,step=1.0,format='%.1f')
        x=st.number_input('Hydrogen mass fraction X',key='guess_x',min_value=0.001,max_value=0.999,step=0.01,format='%.3f')
        z=st.number_input('Metal mass fraction Z',key='guess_z',min_value=0.000001,max_value=0.5,step=0.001,format='%.6f')
        submitted=st.form_submit_button('Run model',type='primary')
    st.caption('Helium fraction Y = 1 − X − Z. Luminosity and temperature are trial boundary conditions, not predictions.')
    st.caption('Initial inputs reproduce the 1 M☉ trial from your notebook.')
if submitted:
    try:
        with st.spinner('Integrating stellar structure…'):
            result=calculate(mass,lum,teff,x,z)
            st.session_state.previous_trial=st.session_state.get('current')
            st.session_state.current=result
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
import html
passed=r['flag']==0 and r['error']==0
headline='Passed the solver core checks' if passed else 'Numerical integration failed' if r['error'] else msg
background='#ecfdf5' if passed else '#fef2f2'
border='#16a34a' if passed else '#dc2626'
text_color='#14532d' if passed else '#7f1d1d'
st.markdown(
    f'<div role="status" style="background:{background};border-left:6px solid {border};'
    f'border-radius:8px;padding:18px 22px;margin:12px 0;color:{text_color};">'
    f'<div style="font-size:25px;font-weight:800;line-height:1.3;margin-bottom:10px;">'
    f'{"✓ " if passed else "Model failed: "}{html.escape(headline)}</div>'
    f'<div style="font-size:18px;line-height:1.5;">{html.escape(explanation(r))}</div></div>',
    unsafe_allow_html=True)
if not passed:
    st.caption(f"Condition flag: {r['flag']} · Integration error: {r['error']}. These profiles describe a failed trial, not an accepted stellar solution.")
with st.expander('What does “Passed” mean? Core checks explained'):
    st.markdown("""The solver integrates inward from the surface. To test whether the remaining inner region can form a consistent core, it first requires:

| Quantity at the last integrated shell | Threshold |
| --- | --- |
| Fractional radius, r/R | Less than 0.02 (2%) |
| Remaining mass, Mᵣ/M★ | Less than 0.01 (1%) |
| Remaining luminosity, Lᵣ/L★ | Less than 0.10 (10%) |

Negative mass or luminosity triggers a separate failure. Reaching the center with too much mass or luminosity remaining also fails.

Once those thresholds are met, the solver checks the extrapolated core:

- **Density:** ρcore = Mᵣ / [(4π/3)r³] must be at least the last shell’s density and no greater than 10 × (ρᵢ/ρᵢ₋₁) × ρᵢ.
- **Energy generation:** εcore = Lᵣ/Mᵣ must be at least the last shell’s energy generation per unit mass.
- **Temperature:** the extrapolated central temperature, calculated using the estimated central pressure and ideal-gas relation, must be at least the last shell’s temperature.

**Passed means all these solver checks succeeded without an integration error.** It is not an independent accuracy guarantee. The remaining luminosity is expected to be generated within the unresolved core; a 10% remainder does not automatically mean a 10% error in total luminosity.

The highlighted remaining mass and luminosity above the plots describe the **last integrated shell**. The plotted points at r=0 use a separate leading-order volume extrapolation of the residuals; those plotted residuals are not used for the solver’s acceptance checks.
""")
st.caption('The residuals below describe the innermost finite shell at positive radius. They are not the mass or luminosity of a point at the center.')
df=r['profile']; inner=diagnostics(r)
cols=st.columns(4)
cols[0].metric('Radius (R☉)',f"{r['radius']/6.9599e10:.4g}")
cols[1].metric('Innermost r/R',f"{inner['r/R']:.4g}")
previous=st.session_state.get('previous_trial')
comparable=previous is not None and tuple(previous['parameters'][i] for i in [0,3,4])==tuple(r['parameters'][i] for i in [0,3,4])
old=diagnostics(previous) if comparable else None
for col,label,key in [(cols[2],'Remaining M/M★','M/M★'),(cols[3],'Remaining L/L★','L/L★')]:
    value=inner[key]
    change=''
    if old is not None and np.isfinite([value,old[key]]).all():
        delta=abs(value)-abs(old[key])
        direction='↓ smaller' if delta<0 else '↑ larger' if delta>0 else 'unchanged'
        color='#15803d' if delta<0 else '#b91c1c' if delta>0 else '#475569'
        change=f'<div style="font-size:14px;color:{color};margin-top:6px;">|Residual|: {direction} ({delta:+.3g}) vs previous trial</div>'
    residual_bg='#ecfdf5' if passed else '#fef2f2'
    residual_border='#bbf7d0' if passed else '#fecaca'
    residual_color='#166534' if passed else '#991b1b'
    with col:
        st.markdown(f'<div style="background:{residual_bg};border:1px solid {residual_border};border-radius:8px;padding:12px;color:{residual_color};">'
                    f'<div style="font-size:14px;">{label}</div><div style="font-size:30px;font-weight:700;">{value:.5g}</div>{change}</div>',unsafe_allow_html=True)
if comparable:
    st.caption(f"Previous trial stopped at r/R={old['r/R']:.4g}; current at {inner['r/R']:.4g}. Smaller absolute residuals alone do not establish improvement when stopping radii differ.")
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

show_points=st.toggle('Show integrated points',value=False)
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
if reference:
    st.caption('Comparison: wide translucent curves. Current model: thin darker curves, drawn on top. When points are enabled, comparison points are open circles and current points are filled circles.')
from plots import make_profiles, invalid_shells
st.plotly_chart(make_profiles(r,reference,coord,axis,log,log_x=log_x,x_range=x_range,show_core=show_core,show_points=show_points),width='stretch')
st.caption('Temperature and density are divided by their own positive maxima; hover to see actual values. Comparison models use their own maxima. Extrapolated core points appear only when enabled.')
if invalid_shells(df).any():
    st.warning('Red shading and solid red lines mark shells with negative radius, mass, luminosity, opacity or energy generation; nonpositive temperature, pressure or density; or nonfinite values. These are invalid computed values, not a physical stellar region. Nonpositive values are omitted on logarithmic axes. On the mass axis, only lines are used because the coordinate can reverse in failed models.')
if r['flag']!=0 or r['error']:
    st.caption('Red inner shading on the radius axis marks the unresolved interior from the last valid shell to the center, not a computed physical region. On a logarithmic axis it extends to the visible lower edge. The dashed red line marks the innermost finite positive-radius shell of the failed integration.')
if reference and (reference['flag']!=0 or reference['error']):
    st.caption('The comparison is also a failed trial. Red diagnostic markers apply to the current model only.')
st.caption('The starting surface shells are assumed radiative; transport labels come from the solver.')
with st.expander('Advanced plots: pressure, opacity and temperature gradient'):
    st.plotly_chart(make_profiles(r,reference,coord,axis,log,advanced=True,log_x=log_x,x_range=x_range,show_core=show_core,show_points=show_points),width='stretch')
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
