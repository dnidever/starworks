"""Coarse luminosity/temperature searches with explicit trial inspection."""
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from model_runner import STATUS,diagnostics
from fast_grid import grid_summary

@st.cache_data(max_entries=30,show_spinner=False)
def cached_grid(mass,x,z,ls,ts):
    return grid_summary(mass,x,z,ls,ts)

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
            nl=a.number_input('Luminosity samples',min_value=3,max_value=200,value=25,step=1)
            nt=b.number_input('Temperature samples',min_value=3,max_value=200,value=25,step=1)
            log_l=st.checkbox('Logarithmic luminosity spacing',value=False)
            run=st.form_submit_button('Run grid search',type='primary')
        if run:
            if low_l>=high_l or low_t>=high_t:
                st.error('Each minimum must be smaller than its maximum.')
            else:
                ls=np.geomspace(low_l,high_l,int(nl)) if log_l else np.linspace(low_l,high_l,int(nl))
                ts=np.linspace(low_t,high_t,int(nt))
                rows=[]
                if not (0<x<1 and 0<z<1 and x+z<1):
                    st.error('Use positive X and Z with X + Z < 1.')
                    return
                with st.spinner('Searching trial models… The first search after a restart may take longer while the solver compiles.'):
                    results=cached_grid(float(mass),float(x),float(z),ls,ts)
                for trial_l,trial_t,flag,error,rr,mm,ll in results:
                    flag=int(flag);error=int(error)
                    rows.append({'L (L☉)':trial_l,'Teff (K)':trial_t,'Accepted':flag==0 and error==0,
                                 'Flag':flag if not error else -2,'r/R':rr,'M/M★':mm,'L/L★':ll,
                                 'Status':STATUS.get(flag,'Unknown') if not error else 'Numerical error'})
                st.session_state.grid_generation=st.session_state.get('grid_generation',0)+1
                st.session_state.grid_result=dict(parameters=(mass,x,z),rows=rows,ls=ls,ts=ts,log_l=log_l)
        grid=st.session_state.get('grid_result')
        if not grid:return
        if grid['parameters']!=(mass,x,z):
            st.info('Mass or composition changed. Run a new grid search for these parameters.')
            return
        data=pd.DataFrame(grid['rows'])
        st.write(f"{int(data.Accepted.sum())} of {len(data)} trials passed the solver core checks.")
        # Colorbar reads bottom to top; place Passed at its highest category.
        flag_order=[-2,-1,1,2,3,4,5,6,0]
        colors=['#64748b','#a78bfa','#eab308','#f97316','#ec4899','#dc2626','#991b1b','#2563eb','#22c55e']
        scale=[]
        for i,color in enumerate(colors):scale.extend([(i/9,color),((i+1)/9,color)])
        matrix=data.Flag.to_numpy().reshape(len(grid['ls']),len(grid['ts']))
        display_matrix=np.vectorize({flag:i for i,flag in enumerate(flag_order)}.__getitem__)(matrix)
        fig=go.Figure(go.Heatmap(x=grid['ts'].tolist(),y=grid['ls'].tolist(),z=display_matrix.tolist(),zmin=-.5,zmax=8.5,colorscale=scale,
            customdata=data.Status.to_numpy().reshape(matrix.shape).tolist(),hovertemplate='Teff=%{x:.2f} K<br>L=%{y:.6g} L☉<br>%{customdata}<extra></extra>',
            colorbar=dict(tickvals=list(range(9)),ticktext=['Numerical error','Shell limit','Density','Energy generation','Temperature','Negative mass','Negative luminosity','Center mismatch','Passed'])))
        fig.update_layout(height=440,xaxis_title='Effective temperature (K)',yaxis_title='Luminosity (L☉)')
        if grid['log_l']:fig.update_yaxes(type='log')
        # Heatmaps do not expose point selection in Streamlit. A transparent
        # scatter layer provides selectable trial centers over the same map.
        fig.add_trace(go.Scatter(x=data['Teff (K)'].tolist(),y=data['L (L☉)'].tolist(),mode='markers',
            marker=dict(symbol='square',size=max(4,min(28,360/max(len(grid['ls']),len(grid['ts']))))),opacity=.01,
            customdata=data.index.tolist(),showlegend=False,hoverinfo='skip',name='Select trial'))
        chart_key=f"grid_map_{st.session_state.get('grid_generation',0)}"
        def choose_grid_point():
            event=st.session_state.get(chart_key,{})
            points=event.get('selection',{}).get('points',[])
            if not points:return
            point=points[-1]
            if point.get('curve_number')!=1:return
            index=int(point['point_index'])
            row=data.iloc[index]
            st.session_state.pending_grid_guess=(mass,float(row['L (L☉)']),float(row['Teff (K)']),x,z)
        st.plotly_chart(fig,width='stretch',key=chart_key,on_select=choose_grid_point,selection_mode='points')
        st.caption('Click a trial center on the map to load its luminosity and temperature into the sidebar as your next guess, then click Run model. The displayed model stays unchanged until you run it.')
        st.caption('Green cells passed the core checks. A coarse grid may miss a narrow solution region: reduce the bounds and search again. Inspect residuals and stopping radius before accepting a model.')
        st.dataframe(data,hide_index=True,width='stretch')
        indices=list(data.index)
        choice=st.selectbox('Trial to inspect',indices,format_func=lambda i:f"{i+1}: L={data.loc[i,'L (L☉)']:.6g}, Teff={data.loc[i,'Teff (K)']:.2f} K — {data.loc[i,'Status']}")
        if st.button('Inspect selected grid trial'):
            row=data.loc[choice]
            try:
                result=calculate(mass,float(row['L (L☉)']),float(row['Teff (K)']),x,z)
                st.session_state.previous_trial=st.session_state.get('current')
                st.session_state.current=result
                st.rerun()
            except Exception as exc:st.error(f'Could not inspect trial: {exc}')
        st.download_button('Download grid results (CSV)',data.to_csv(index=False),'starworks_grid.csv','text/csv')
