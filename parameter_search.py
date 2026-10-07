"""One-dimensional parameter sweeps using the compiled summary solver."""
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st
from grid_search import cached_grid
from model_runner import STATUS
from solver_version import SOLVER_REVISION


def parameter_search_ui(calculate,current):
    mass,lum,temp,x,z=current['parameters']
    st.subheader('One-dimensional parameter search')
    st.caption(f'Baseline: M={mass:g} M☉ · L={lum:.8g} L☉ · Teff={temp:.6f} K · X={x:g} · Z={z:g}')
    st.write('First sweep luminosity at fixed temperature, then sweep temperature at fixed luminosity. Each search changes one input so you can see how the two residuals respond.')
    context=str(current['parameters'])
    for axis in ['Luminosity','Temperature']:
        is_l=axis=='Luminosity'
        center=lum if is_l else temp
        unit='L☉' if is_l else 'K'
        key=f'sweep_{axis}'
        st.subheader(f'{1 if is_l else 2}. Search {axis.lower()}')
        with st.form(f'{key}_controls'):
            a,b,c=st.columns(3)
            low=a.number_input(f'Minimum {axis.lower()} ({unit})',min_value=.000001 if is_l else 1.,value=float(center*.9),format='%.8f',key=f'{key}_low_{context}')
            high=b.number_input(f'Maximum {axis.lower()} ({unit})',min_value=.000001 if is_l else 1.,value=float(center*1.1),format='%.8f',key=f'{key}_high_{context}')
            count=c.number_input('Samples',min_value=3,max_value=500,value=51,key=f'{key}_samples')
            fixed=st.number_input('Fixed temperature (K)' if is_l else 'Fixed luminosity (L☉)',min_value=1. if is_l else .000001,value=float(temp if is_l else lum),format='%.8f',key=f'{key}_fixed_{context}')
            run=st.form_submit_button(f'Run {axis.lower()} search',type='primary')
        if run:
            if low>=high:st.error('Minimum must be smaller than maximum.')
            else:
                values=np.linspace(low,high,int(count))
                ls=values if is_l else np.array([fixed])
                ts=np.array([fixed]) if is_l else values
                with st.spinner(f'Searching {axis.lower()}…'):
                    rows=cached_grid(float(mass),float(x),float(z),ls,ts,SOLVER_REVISION)
                data=pd.DataFrame(rows,columns=['L (L☉)','Teff (K)','Flag','Error','r/R','M/M★','L/L★','Density / minimum','Density / maximum','Core / shell energy','Core / shell temperature'])
                data['Status']=[STATUS.get(int(f),'Unknown') if not e else 'Unphysical pressure, temperature, or density' for f,e in zip(data.Flag,data.Error)]
                data['Passed']=(data.Flag==0)&(data.Error==0)
                st.session_state[key]=dict(context=context,data=data)
        result=st.session_state.get(key)
        if not result or result['context']!=context:continue
        data=result['data']
        horizontal='L (L☉)' if is_l else 'Teff (K)'
        fig=make_subplots(rows=1,cols=2,subplot_titles=['Remaining mass / stellar mass','Remaining luminosity / stellar luminosity'])
        custom=data[['Status','r/R','L (L☉)','Teff (K)']].values.tolist()
        for col,(quantity,limit,color) in enumerate([('M/M★',.01,'#b45309'),('L/L★',.1,'#0369a1')],1):
            # Errors interrupt the line; hoverable error markers retain finite diagnostics.
            valid=(data.Error==0)&(data.Flag>=0)
            fig.add_trace(go.Scatter(x=data[horizontal].tolist(),y=data[quantity].where(valid,np.nan).tolist(),mode='lines+markers',line=dict(color=color),marker=dict(size=6),customdata=custom,
                hovertemplate=f'{axis}=%{{x:.8g}} {unit}<br>{quantity}=%{{y:.6g}}<br>%{{customdata[0]}}<br>Last r/R=%{{customdata[1]:.6g}}<br>L=%{{customdata[2]:.8g}} L☉<br>Teff=%{{customdata[3]:.8g}} K<extra></extra>',showlegend=False),row=1,col=col)
            passed=data.Passed
            fig.add_trace(go.Scatter(x=data.loc[passed,horizontal].tolist(),y=data.loc[passed,quantity].tolist(),mode='markers',marker=dict(size=10,color='#16a34a',symbol='diamond'),name='Passed all core checks',showlegend=col==1),row=1,col=col)
            errors=~valid&np.isfinite(data[quantity])
            fig.add_trace(go.Scatter(x=data.loc[errors,horizontal].tolist(),y=data.loc[errors,quantity].tolist(),mode='markers',marker=dict(size=8,color='#64748b',symbol='x'),customdata=[custom[i] for i in np.flatnonzero(errors)],hovertemplate='%{customdata[0]}<br>Residual=%{y:.6g}<br>Last r/R=%{customdata[1]:.6g}<extra></extra>',name='Integration error or limit',showlegend=col==1),row=1,col=col)
            fig.add_hrect(y0=0,y1=limit,fillcolor='#22c55e',opacity=.1,line_width=0,row=1,col=col)
            fig.add_hline(y=0,line_color='#64748b',row=1,col=col)
            fig.add_hline(y=limit,line_dash='dot',line_color='#16a34a',row=1,col=col)
            fig.add_vline(x=center,line_dash='dot',line_color='#94a3b8',row=1,col=col)
            fig.update_xaxes(title_text=f'{axis} ({unit})',row=1,col=col)
        fig.update_layout(height=420,legend=dict(orientation='h',y=1.18))
        st.plotly_chart(fig,width='stretch',key=f'{key}_plot')
        st.caption('Green shading marks each residual’s allowed range; green diamonds pass all core checks. Gray crosses are incomplete integrations. The vertical dotted line is the baseline input. Hover to compare stopping radii: changes in stopping radius can affect residual trends.')
        with st.expander(f'{axis} search results'):
            st.dataframe(data,hide_index=True,width='stretch')
        selected=st.selectbox(f'{axis} trial to run',data.index.tolist(),format_func=lambda i,data=data,horizontal=horizontal,unit=unit:f"{data.loc[i,horizontal]:.8g} {unit} — {data.loc[i,'Status']}",key=f'{key}_choice')
        if st.button(f'Run selected {axis.lower()} trial'):
            row=data.loc[selected]
            model=calculate(mass,float(row['L (L☉)']),float(row['Teff (K)']),x,z)
            st.session_state.previous_trial=current
            st.session_state.current=model
            st.session_state.pending_grid_guess=model['parameters']
            st.session_state.switch_to_model_tab=True
            st.rerun()
