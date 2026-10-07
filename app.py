import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st
from model_runner import run_model, STATUS, diagnostics, explanation
st.set_page_config(page_title='StarWorks',page_icon='☀️',layout='wide')
st.title('StarWorks')
st.caption('Explore homogeneous main-sequence models with STATSTAR. Adjust the surface conditions, integrate inward, and inspect the interior.')
from solver_version import SOLVER_REVISION
@st.cache_data(max_entries=200,show_spinner=False)
def cached_calculate(pars,solver_revision):
    return run_model(*pars)
def calculate(*pars):
    return cached_calculate(pars,SOLVER_REVISION)
if st.session_state.get('solver_revision')!=SOLVER_REVISION:
    had_results='current' in st.session_state or 'grid_result' in st.session_state
    for key in ['current','previous_trial','reference','grid_result','direction_tests','pending_grid_guess']:
        st.session_state.pop(key,None)
    st.session_state.solver_revision=SOLVER_REVISION
    if had_results:
        st.info('The solver was updated. Previous model and grid results were cleared; run your trial or grid again.')
for key,value in {'guess_mass':1.0,'guess_lum':0.8652,'guess_teff':5513.5,'guess_x':.70,'guess_z':.008}.items():
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
        teff=st.number_input('Effective temperature (K)',key='guess_teff',min_value=1.0,step=0.01,format='%.6f')
        x=st.number_input('Hydrogen mass fraction X',key='guess_x',min_value=0.001,max_value=0.999,step=0.01,format='%.3f')
        z=st.number_input('Metal mass fraction Z',key='guess_z',min_value=0.000001,max_value=0.5,step=0.001,format='%.6f')
        submitted=st.form_submit_button('Run model',type='primary')
    st.caption('Helium fraction Y = 1 − X − Z. Luminosity and temperature are trial boundary conditions, not predictions.')
    st.caption('Initial inputs give a verified passing 1 M☉ trial at X=0.70 and Z=0.008.')
if submitted:
    try:
        with st.spinner('Integrating stellar structure…'):
            result=calculate(mass,lum,teff,x,z)
            st.session_state.previous_trial=st.session_state.get('current')
            st.session_state.current=result
    except Exception as exc:
        st.error(f'Model failed: {exc}')
st.markdown("""
<style>
.stTabs [data-baseweb="tab-list"] {
    gap: 12px; padding: 8px 0 14px; flex-wrap: wrap;
}
.stTabs [data-baseweb="tab"] {
    height: 58px; padding: 10px 24px; border: 2px solid #64748b !important;
    border-radius: 18px !important; background: #f1f5f9 !important;
    color: #334155 !important; box-shadow: 0 2px 4px rgba(0,0,0,0.10);
}
.stTabs [data-baseweb="tab"] p {
    font-size: 20px; font-weight: 700;
}
.stTabs [data-baseweb="tab"][aria-selected="true"] {
    background: #dbeafe !important; color: #1e3a8a !important; border: 3px solid #2563eb !important;
    box-shadow: 0 2px 5px rgba(37,99,235,0.15);
}
.stTabs [data-baseweb="tab"]:hover { border-color: #2563eb; }
.stTabs [data-baseweb="tab-highlight"] { display: none; }
</style>
""",unsafe_allow_html=True)
model_tab,adjustment_tab,grid_tab=st.tabs(['☀️ Model','↔️ Adjustment tests','▦ Grid search'])
from grid_search import grid_search_ui
with grid_tab:
    grid_search_ui(calculate,(mass,x,z),st.session_state.get('current'))
if 'current' not in st.session_state:
    with model_tab:
        st.info('Choose sidebar parameters and click Run model, or select a passing trial in the Grid search tab.')
    with adjustment_tab:
        st.info('Run a model first, then use this tab to compare nearby luminosity and temperature adjustments.')
    st.stop()
with model_tab:
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
    st.subheader('Last finite-radius shell values and core density check')
    st.caption('The residuals below describe the innermost finite shell at positive radius. They are not the mass or luminosity of a point at the center.')
    df=r['profile']; inner=diagnostics(r)
    cols=st.columns(4)
    cols[0].metric('Innermost r/R',f"{inner['r/R']:.4g}")
    previous=st.session_state.get('previous_trial')
    comparable=previous is not None and tuple(previous['parameters'][i] for i in [0,3,4])==tuple(r['parameters'][i] for i in [0,3,4])
    old=diagnostics(previous) if comparable else None
    for col,label,key in [(cols[1],'Remaining M/M★','M/M★'),(cols[2],'Remaining L/L★','L/L★')]:
        value=inner[key]
        change=''
        if old is not None and np.isfinite([value,old[key]]).all():
            delta=abs(value)-abs(old[key])
            direction='↓ smaller' if delta<0 else '↑ larger' if delta>0 else 'unchanged'
            color='#15803d' if delta<0 else '#b91c1c' if delta>0 else '#475569'
            change=f'<div style="font-size:14px;color:{color};margin-top:6px;">|Residual|: {direction} ({delta:+.3g}) vs previous trial</div>'
        limit=.01 if key=='M/M★' else .10
        individual_ok=np.isfinite(value) and 0<=value<limit
        change+=f'<div style="font-size:13px;margin-top:5px;">{"✓" if individual_ok else "✗"} Required: 0 ≤ value &lt; {limit:g}</div>'
        residual_bg='#ecfdf5' if individual_ok else '#fef2f2'
        residual_border='#bbf7d0' if individual_ok else '#fecaca'
        residual_color='#166534' if individual_ok else '#991b1b'
        with col:
            st.markdown(f'<div style="background:{residual_bg};border:1px solid {residual_border};border-radius:8px;padding:12px;color:{residual_color};">'
                        f'<div style="font-size:14px;">{label}</div><div style="font-size:30px;font-weight:700;">{value:.5g}</div>{change}</div>',unsafe_allow_html=True)
    density_shells=df[(df.r>0)&np.isfinite(df.rho)&(df.rho>0)]
    with cols[3]:
        if len(density_shells)>=2:
            rho_last=float(density_shells.iloc[0].rho)
            rho_upper=10*rho_last*rho_last/float(density_shells.iloc[1].rho)
            rho_core=float(r['core']['rho'])
            density_ok=np.isfinite(rho_core) and rho_last<=rho_core<=rho_upper
            bg='#ecfdf5' if density_ok else '#fef2f2'
            color='#166534' if density_ok else '#991b1b'
            border='#bbf7d0' if density_ok else '#fecaca'
            st.markdown(f'<div style="background:{bg};border:1px solid {border};border-radius:8px;padding:12px;color:{color};">'
                f'<div style="font-size:14px;">Core density (g/cm³)</div><div style="font-size:30px;font-weight:700;">{rho_core:.5g}</div>'
                f'<div style="font-size:13px;margin-top:5px;">Last shell: {rho_last:.5g}<br>{"✓" if density_ok else "✗"} Allowed: {rho_last:.5g}–{rho_upper:.5g}</div></div>',unsafe_allow_html=True)
        else:
            st.metric('Core density check','Unavailable')
    st.caption('Each box evaluates its own criterion; green individual checks do not by themselves establish that the full model passes.')
    if comparable:
        st.caption(f"Previous trial stopped at r/R={old['r/R']:.4g}; current at {inner['r/R']:.4g}. Smaller absolute residuals alone do not establish improvement when stopping radii differ.")

with adjustment_tab:
    st.subheader('Test adjustment directions')
    st.caption(f'Current trial: M={m:g} M☉ · L={l:.8g} L☉ · Teff={t:.6f} K · X={h:g} · Z={met:g}')
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


with model_tab:
    from plots import make_profiles, invalid_shells
    with st.expander('Plot controls',expanded=True):
        axis_col,y_col,x_col=st.columns([2,1,1])
        with axis_col:
            axis=st.radio('Horizontal axis',['Fractional radius','Enclosed mass fraction'],horizontal=True)
        log=y_col.checkbox('Logarithmic positive profiles',value=True)
        log_x=x_col.toggle('Logarithmic x-axis',value=False)
        coord='r_fraction' if axis=='Fractional radius' else 'm_fraction'
        # Reset editable bounds only for a newly displayed model or coordinate scale.
        range_context=(r['parameters'],coord,log_x,r['flag'],r['error'])
        if st.session_state.get('range_context')!=range_context:
            st.session_state.range_context=range_context
            st.session_state.limit_plot_range=not passed
            valid=df[(df.r>0)&~invalid_shells(df)&np.isfinite(df[coord])]
            if log_x:valid=valid[valid[coord]>0]
            boundary=float(valid.iloc[0][coord]) if len(valid) else .03
            maximum=min(.3,10*boundary) if boundary>0 else .3
            minimum=max(boundary*.01,1e-12) if log_x else 0.0
            st.session_state.plot_xmin=minimum
            st.session_state.plot_xmax=maximum if maximum>minimum else .3
        climit,cpoints,ccore=st.columns(3)
        manual=climit.toggle('Limit x-range',key='limit_plot_range')
        show_points=cpoints.toggle('Show integrated points',value=False)
        show_core=ccore.toggle('Show extrapolated core point',value=True)
        coord='r_fraction' if axis=='Fractional radius' else 'm_fraction'
        x_range=None
        if manual:
            cmin,cmax=st.columns(2)
            xmin=cmin.number_input('Minimum x',key='plot_xmin',format='%.6f')
            xmax=cmax.number_input('Maximum x',key='plot_xmax',format='%.6f')
            if not np.isfinite([xmin,xmax]).all() or xmin>=xmax or (log_x and xmin<=0):
                st.error('Use minimum < maximum, with positive bounds for a logarithmic x-axis.')
            else:x_range=(xmin,xmax)
        ckeep,cclear=st.columns(2)
        reference=st.session_state.get('reference')
        if ckeep.button('Keep current model as comparison'):
            st.session_state.reference=r
            reference=r
        if cclear.button('Clear comparison'):
            st.session_state.pop('reference',None)
            reference=None
        if log_x:st.caption('Logarithmic axes omit zero and negative coordinates, including the zero-radius core point.')
        if show_core:st.caption('Open diamonds show extrapolated core values. Core mass and luminosity are approximate residuals, not integrated points.')
        if reference:st.caption('Comparison curves are wide and translucent; current curves are thin and dark.')
    st.plotly_chart(make_profiles(r,reference,coord,axis,log,log_x=log_x,x_range=x_range,show_core=show_core,show_points=show_points),width='stretch')
    if show_core and any(r['core'][q]<=0 for q in ['T','P','epsilon']):
        st.caption('Nonpositive extrapolated core temperature, pressure or energy generation is omitted from these plots because that core estimate is unphysical. Values remain available under numerical diagnostics. Negative mass, luminosity and density diagnostics remain visible.')
    if show_core and r['core']['rho']<=0:
        st.caption('The density panel uses a linear y-axis to show the nonpositive inferred core density in red. Turn off logarithmic x scaling to see the core at r=0.')
    st.caption('Temperature and pressure are divided by their own positive maxima; hover to see actual values. Comparison models use their own maxima. Extrapolated core points appear only when enabled.')
    if invalid_shells(df).any():
        st.warning('Red shading and solid red lines mark shells with negative radius, mass, luminosity, opacity or energy generation; nonpositive temperature, pressure or density; or nonfinite values. These are invalid computed values, not a physical stellar region. Nonpositive values are omitted on logarithmic axes. On the mass axis, only lines are used because the coordinate can reverse in failed models.')
    if r['flag']!=0 or r['error']:
        st.caption('Red inner shading on the radius axis marks the unresolved interior from the last valid shell to the center, not a computed physical region. On a logarithmic axis it extends to the visible lower edge. The dashed red line marks the innermost finite positive-radius shell of the failed integration.')
    if reference and (reference['flag']!=0 or reference['error']):
        st.caption('The comparison is also a failed trial. Red diagnostic markers apply to the current model only.')
    st.caption('The starting surface shells are assumed radiative; transport labels come from the solver.')
    with st.expander('Advanced plots: pressure, opacity, temperature gradient and energy transport'):
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
