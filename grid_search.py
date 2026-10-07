"""Coarse luminosity/temperature searches with explicit trial inspection."""
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from model_runner import STATUS,diagnostics

def grid_search_ui(calculate, sidebar_parameters, current):
    with st.expander('Grid search in luminosity and temperature',expanded=current is None):
        mass,x,z=sidebar_parameters
        if current:
            mass,_,_,x,z=current['parameters']
        center_l=current['parameters'][1] if current else .86071
        center_t=current['parameters'][2] if current else 5500.2
        st.write(f'Search at fixed mass {mass:g} M☉, X={x:g}, Z={z:g}. '+('These are the displayed model’s parameters.' if current else 'These are the sidebar mass and composition.'))
        with st.form('grid_controls'):
            a,b=st.columns(2)
            low_l=a.number_input('Minimum luminosity (L☉)',min_value=.000001,value=float(center_l*.8),format='%.6f')
            high_l=b.number_input('Maximum luminosity (L☉)',min_value=.000001,value=float(center_l*1.2),format='%.6f')
            low_t=a.number_input('Minimum temperature (K)',min_value=1.0,value=float(center_t*.9))
            high_t=b.number_input('Maximum temperature (K)',min_value=1.0,value=float(center_t*1.1))
            nl=a.number_input('Luminosity samples',min_value=3,max_value=30,value=15,step=1)
            nt=b.number_input('Temperature samples',min_value=3,max_value=30,value=15,step=1)
            log_l=st.checkbox('Logarithmic luminosity spacing',value=False)
            run=st.form_submit_button('Run grid search',type='primary')
        if run:
            if low_l>=high_l or low_t>=high_t:
                st.error('Each minimum must be smaller than its maximum.')
            else:
                ls=np.geomspace(low_l,high_l,int(nl)) if log_l else np.linspace(low_l,high_l,int(nl))
                ts=np.linspace(low_t,high_t,int(nt))
                rows=[]; progress=st.progress(0,text='Searching trial models…')
                for lum in ls:
                    for temp in ts:
                        row={'L (L☉)':float(lum),'Teff (K)':float(temp),'Accepted':False,'Flag':-2}
                        try:
                            result=calculate(mass,float(lum),float(temp),x,z)
                            row.update(diagnostics(result))
                            row['Accepted']=result['flag']==0 and result['error']==0
                            row['Flag']=result['flag'] if not result['error'] else -2
                            row['Status']=STATUS.get(result['flag'],'Unknown') if not result['error'] else 'Numerical error'
                        except Exception as exc:
                            row['Status']=f'Calculation failed: {exc}'
                        rows.append(row)
                        progress.progress(len(rows)/(len(ls)*len(ts)),text=f'Searched {len(rows)} / {len(ls)*len(ts)} trials')
                progress.empty()
                st.session_state.grid_result=dict(parameters=(mass,x,z),rows=rows,ls=ls,ts=ts,log_l=log_l)
        grid=st.session_state.get('grid_result')
        if not grid:return
        if grid['parameters']!=(mass,x,z):
            st.info('Mass or composition changed. Run a new grid search for these parameters.')
            return
        data=pd.DataFrame(grid['rows'])
        st.write(f"{int(data.Accepted.sum())} of {len(data)} trials passed the solver core checks.")
        colors=['#64748b','#a78bfa','#22c55e','#eab308','#f97316','#ec4899','#dc2626','#991b1b','#2563eb']
        scale=[]
        for i,color in enumerate(colors):scale.extend([(i/9,color),((i+1)/9,color)])
        matrix=data.Flag.to_numpy().reshape(len(grid['ls']),len(grid['ts']))
        fig=go.Figure(go.Heatmap(x=grid['ts'],y=grid['ls'],z=matrix,zmin=-2.5,zmax=6.5,colorscale=scale,
            customdata=data.Status.to_numpy().reshape(matrix.shape),hovertemplate='Teff=%{x:.2f} K<br>L=%{y:.6g} L☉<br>%{customdata}<extra></extra>',
            colorbar=dict(tickvals=list(range(-2,7)),ticktext=['Numerical error','Shell limit','Passed','Density','Energy generation','Temperature','Negative mass','Negative luminosity','Center mismatch'])))
        fig.update_layout(height=440,xaxis_title='Effective temperature (K)',yaxis_title='Luminosity (L☉)')
        if grid['log_l']:fig.update_yaxes(type='log')
        st.plotly_chart(fig,width='stretch')
        st.caption('Green cells passed the core checks. A coarse grid may miss a narrow solution region: reduce the bounds and search again. Inspect residuals and stopping radius before accepting a model.')
        st.dataframe(data,hide_index=True,width='stretch')
        indices=list(data.index)
        choice=st.selectbox('Trial to inspect',indices,format_func=lambda i:f"{i+1}: L={data.loc[i,'L (L☉)']:.6g}, Teff={data.loc[i,'Teff (K)']:.2f} K — {data.loc[i,'Status']}")
        if st.button('Inspect selected grid trial'):
            row=data.loc[choice]
            try:
                st.session_state.current=calculate(mass,float(row['L (L☉)']),float(row['Teff (K)']),x,z)
                st.rerun()
            except Exception as exc:st.error(f'Could not inspect trial: {exc}')
        st.download_button('Download grid results (CSV)',data.to_csv(index=False),'starworks_grid.csv','text/csv')
