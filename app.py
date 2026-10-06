import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st
from model_runner import run_model, STATUS
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
df=r['profile']; inner=df.iloc[0]
cols=st.columns(4)
for col,label,value in zip(cols,['Radius (R☉)','Innermost r/R','Remaining M/M★','Remaining L/L★'],[r['radius']/6.9599e10,inner.r_fraction,inner.m_fraction,inner.l_fraction]):
    col.metric(label,f'{value:.4g}')
a,b=st.columns([3,1])
with a:
    axis=st.radio('Horizontal axis',['Fractional radius','Enclosed mass fraction'],horizontal=True)
with b:
    log=st.checkbox('Logarithmic positive profiles',value=True)
coord='r_fraction' if axis=='Fractional radius' else 'm_fraction'
reference=st.session_state.get('reference')
c1,c2=st.columns(2)
if c1.button('Keep current model as comparison'):
    st.session_state.reference=r
    reference=r
if c2.button('Clear comparison'):
    st.session_state.pop('reference',None)
    reference=None
quantities=[('T','Temperature (K)'),('rho','Density (g cm⁻³)'),('P','Pressure (dyn cm⁻²)'),('m_fraction','Enclosed mass / total mass'),('l_fraction','Luminosity / total luminosity'),('epsilon','Energy generation (erg g⁻¹ s⁻¹)'),('kappa','Opacity (cm² g⁻¹)'),('dlnPdlnT','d ln P / d ln T')]
fig=make_subplots(rows=4,cols=2,subplot_titles=[q[1] for q in quantities],vertical_spacing=0.07)
for i,(q,label) in enumerate(quantities):
    row,col=divmod(i,2); row+=1; col+=1
    for model,name,color,dash in [(r,'Current','#f59e0b','solid'),(reference,'Comparison','#38bdf8','dash')]:
        if model is None: continue
        data=model['profile']; valid=np.isfinite(data[coord]) & np.isfinite(data[q])
        if log and q in ['T','rho','P','epsilon','kappa']: valid &= data[q]>0
        fig.add_trace(go.Scatter(x=data.loc[valid,coord],y=data.loc[valid,q],name=name,legendgroup=name,showlegend=i==0,line=dict(color=color,dash=dash)),row=row,col=col)
    fig.update_xaxes(title_text=axis,row=row,col=col)
    if log and q in ['T','rho','P','epsilon','kappa']: fig.update_yaxes(type='log',row=row,col=col)
fig.update_layout(height=1150,margin=dict(t=55,b=30),hovermode='x unified')
st.plotly_chart(fig,width='stretch')
st.subheader('Energy transport')
transport=go.Figure(go.Scatter(x=df[coord],y=(df.zone=='c').astype(int),mode='lines',line=dict(shape='hv',color='#f59e0b')))
transport.update_layout(height=230,xaxis_title=axis,yaxis=dict(tickvals=[0,1],ticktext=['Radiative','Convective'],range=[-.1,1.1]))
st.plotly_chart(transport,width='stretch')
st.caption('Transport classifications come from the solver. The starting surface shells are assumed radiative. Extrapolated central points are excluded from all profile plots.')
with st.expander('Extrapolated core and numerical diagnostics'):
    st.write('These values extrapolate from the last integrated shell. Central mass and luminosity below are residuals, not physical point values.')
    st.json({k:r['core'][k] for k in ['T','rho','P','epsilon','M','L']})
    st.write({'condition_flag':r['flag'],'integration_error':r['error'],'integrated_shells':len(df)})
    if r['warnings']: st.write(r['warnings'])
with st.expander('Model table and downloads'):
    st.dataframe(df,width='stretch')
    st.download_button('Download integrated shells (CSV)',df.to_csv(index=False),'starmodel_shells.csv','text/csv')
with st.expander('How to use this in class'):
    st.markdown('Keep a model as a comparison, then change one parameter. Examine where the luminosity rises, where energy transport is convective, and how concentrated the mass is. Adjust luminosity and effective temperature to reduce the remaining central mass and luminosity while meeting the core checks. This simplified solver assumes homogeneous composition and is not an evolutionary model.')
