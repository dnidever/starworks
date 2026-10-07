"""Coarse luminosity/temperature searches with explicit trial inspection."""
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from model_runner import STATUS,diagnostics
from fast_grid import grid_summary
from solver_version import SOLVER_REVISION

@st.cache_data(max_entries=30,show_spinner=False)
def cached_grid(mass,x,z,ls,ts,solver_revision):
    return grid_summary(mass,x,z,ls,ts)

def grid_search_ui(calculate, sidebar_parameters, current):
    st.subheader('Grid search')
    with st.expander('Search luminosity and temperature',expanded=True):
        mass,x,z=sidebar_parameters
        if current:
            mass,_,_,x,z=current['parameters']
        center_l=current['parameters'][1] if current else .8652
        center_t=current['parameters'][2] if current else 5513.5
        st.write(f'Search at fixed mass {mass:g} M☉, X={x:g}, Z={z:g}. '+('These are the displayed model’s parameters.' if current else 'These are the sidebar mass and composition.'))
        for key,value in zip(['grid_low_l','grid_high_l','grid_low_t','grid_high_t'],
                             [center_l*.98,center_l*1.02,center_t*.99,center_t*1.01]):
            st.session_state.setdefault(key,float(value))
        pending=st.session_state.pop('pending_search_bounds',None)
        if pending:
            for key,value in zip(['grid_low_l','grid_high_l','grid_low_t','grid_high_t'],pending):
                st.session_state[key]=float(value)
        with st.form('grid_controls'):
            a,b=st.columns(2)
            low_l=a.number_input('Minimum luminosity (L☉)',key='grid_low_l',min_value=.000001,value=None,step=0.02,format='%.6f')
            high_l=b.number_input('Maximum luminosity (L☉)',key='grid_high_l',min_value=.000001,value=None,step=0.02,format='%.6f')
            low_t=a.number_input('Minimum temperature (K)',key='grid_low_t',min_value=1.0,value=None,step=10.0)
            high_t=b.number_input('Maximum temperature (K)',key='grid_high_t',min_value=1.0,value=None,step=10.0)
            nl=a.number_input('Luminosity samples',min_value=3,max_value=200,value=50,step=1)
            nt=b.number_input('Temperature samples',min_value=3,max_value=200,value=50,step=1)
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
                    results=cached_grid(float(mass),float(x),float(z),ls,ts,SOLVER_REVISION)
                for trial_l,trial_t,flag,error,rr,mm,ll in results:
                    flag=int(flag);error=int(error)
                    rows.append({'L (L☉)':trial_l,'Teff (K)':trial_t,'Accepted':flag==0 and error==0,
                                 'Flag':flag if not error else -2,'r/R':rr,'M/M★':mm,'L/L★':ll,
                                 'Status':STATUS.get(flag,'Unknown') if not error else 'Unphysical pressure, temperature, or density'})
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
        status_labels=['Unphysical pressure, temperature, or density','Integration limit reached',
                       'Core density outside allowed range','Core energy generation too low',
                       'Core temperature too low','Enclosed mass became negative',
                       'Luminosity became negative','Too much mass or luminosity near center','Passed']
        data['Status']=data.Flag.map(dict(zip(flag_order,status_labels)))
        colors=['#64748b','#a78bfa','#eab308','#f97316','#ec4899','#dc2626','#991b1b','#2563eb','#22c55e']
        scale=[]
        for i,color in enumerate(colors):scale.extend([(i/9,color),((i+1)/9,color)])
        matrix=data.Flag.to_numpy().reshape(len(grid['ls']),len(grid['ts']))
        display_matrix=np.vectorize({flag:i for i,flag in enumerate(flag_order)}.__getitem__)(matrix)
        # A residual heuristic, not a measurement of all core mismatches.
        residual=np.maximum.reduce([np.abs(data['M/M★'].to_numpy())/.01,
                                    np.abs(data['L/L★'].to_numpy())/.1,
                                    np.abs(data['r/R'].to_numpy())/.02])
        eligible=(data.Flag>=0)&np.isfinite(residual)
        score=np.where(eligible,1/(1+residual),np.nan)
        score=np.where(data.Accepted,1.0,np.minimum(score,.99))
        data['Promise score']=score
        map_quantity=st.selectbox('Map coloring',[
            'Model status','Promising models','Remaining mass (M/M★)',
            'Remaining luminosity (L/L★)','Last finite shell radius (r/R)'],
            key='grid_map_quantity')
        hover='Teff=%{x:.6f} K<br>L=%{y:.8g} L☉<br>%{customdata[0]}<br>Last shell r/R=%{customdata[1]:.5g}<br>Remaining M/M★=%{customdata[2]:.5g}<br>Remaining L/L★=%{customdata[3]:.5g}<extra></extra>'
        details=data[['Status','r/R','M/M★','L/L★']].values
        if map_quantity=='Model status':
            heatmap_options=dict(z=display_matrix.tolist(),zmin=-.5,zmax=8.5,colorscale=scale,
                colorbar=dict(tickvals=list(range(9)),ticktext=status_labels))
        elif map_quantity=='Promising models':
            heatmap_options=dict(z=score.reshape(matrix.shape).tolist(),zmin=0,zmax=1,
                colorscale='Viridis',colorbar=dict(title=dict(text='Promise score')))
        else:
            column={'Remaining mass (M/M★)':'M/M★',
                    'Remaining luminosity (L/L★)':'L/L★',
                    'Last finite shell radius (r/R)':'r/R'}[map_quantity]
            values=data[column].to_numpy(dtype=float)
            values=np.where(np.isfinite(values),values,np.nan)
            heatmap_options=dict(z=values.reshape(matrix.shape).tolist(),colorscale='Viridis',
                                 colorbar=dict(title=dict(text=column)))
            if column!='r/R' and np.any(values<0):
                extent=float(np.nanmax(np.abs(values)))
                heatmap_options.update(colorscale='RdBu',zmin=-extent,zmax=extent,zmid=0)
        fig=go.Figure(go.Heatmap(x=grid['ts'].tolist(),y=grid['ls'].tolist(),
            customdata=details.reshape(*matrix.shape,4).tolist(),hovertemplate=hover,
            **heatmap_options))
        fig.update_layout(height=560,xaxis_title='Effective temperature (K)',yaxis_title='Luminosity (L☉)')
        if grid['log_l']:fig.update_yaxes(type='log')
        # Heatmaps do not expose point selection in Streamlit. A transparent
        # scatter layer provides selectable trial centers over the same map.
        fig.add_trace(go.Scatter(x=data['Teff (K)'].tolist(),y=data['L (L☉)'].tolist(),mode='markers',
            marker=dict(symbol='square',size=max(4,min(28,360/max(len(grid['ls']),len(grid['ts']))))),opacity=.05,
            customdata=data[['Status','r/R','M/M★','L/L★']].values.tolist(),showlegend=False,
            hovertemplate=hover,name='Select trial'))
        fig.update_layout(clickmode='event+select',dragmode=False)
        chart_key=f"grid_map_{st.session_state.get('grid_generation',0)}"
        def choose_grid_point():
            event=st.session_state.get(chart_key,{})
            points=event.get('selection',{}).get('points',[])
            if not points:return
            point=points[-1]
            if point.get('curve_number',1)!=1:return
            index=int(point.get('point_index',point.get('point_number',-1)))
            if not 0<=index<len(data):return
            row=data.iloc[index]
            st.session_state.grid_focus_index=index
            st.session_state.pending_grid_guess=(mass,float(row['L (L☉)']),float(row['Teff (K)']),x,z)
        _,map_column,_=st.columns([1,3,1])
        with map_column:
            st.plotly_chart(fig,width='stretch',key=chart_key,on_select=choose_grid_point,selection_mode='points')
        st.caption('Click a trial center on the map to load its luminosity and temperature into the sidebar as your next guess, then click Run model. The displayed model stays unchanged until you run it.')
        if map_quantity=='Promising models':
            st.caption('Higher is more promising: the score uses the largest absolute mass, luminosity, or radius residual divided by its core threshold. Passing models score 1; failures remain below 1. Numerical errors and integration limits are blank. Density, energy generation, and temperature mismatch sizes are not included, so this is guidance, not an acceptance test.')
        elif map_quantity!='Model status':
            st.caption('Colors show the signed last finite shell value as a fraction of the total stellar mass, luminosity, or radius. Missing values appear as gaps. Hover to see each model’s status and diagnostics.')
        st.caption('In the Model status view, green cells passed the core checks. A coarse grid may miss a narrow solution region: reduce the bounds and search again. Inspect residuals and stopping radius before accepting a model.')
        candidates=data.loc[eligible].sort_values(['Accepted','Promise score'],ascending=False)
        if len(candidates):
            best=candidates.iloc[0]
            directions=[]
            for col,axis,values in [('Teff (K)','temperature',grid['ts']),('L (L☉)','luminosity',grid['ls'])]:
                fraction=(best[col]-values[0])/(values[-1]-values[0])
                if fraction<=.1:directions.append(f'extend {axis} downward')
                elif fraction>=.9:directions.append(f'extend {axis} upward')
            advice=('Try '+', and '.join(directions)+'.') if directions else 'Try a finer search around this trial within the current bounds.'
            st.info(f"{'A passing' if best.Accepted else 'The highest-ranked non-error'} trial is L={best['L (L☉)']:.8g} L☉, Teff={best['Teff (K)']:.6f} K. {advice} This suggestion follows the sampled results; it does not establish a unique direction toward a solution.")
        else:
            st.info('No trials provide usable residual guidance. Try bounds around a model that integrates successfully.')
        focus_options=candidates.index.tolist()+[i for i in data.index if i not in candidates.index]
        focus_key=f"grid_focus_{st.session_state.get('grid_generation',0)}"
        clicked=st.session_state.pop('grid_focus_index',None)
        if clicked in focus_options:st.session_state[focus_key]=clicked
        focus=st.selectbox('Trial to center the next search on',focus_options,key=focus_key,
            format_func=lambda i:f"L={data.loc[i,'L (L☉)']:.8g} L☉, Teff={data.loc[i,'Teff (K)']:.6f} K — {data.loc[i,'Status']}")
        if st.button('Search around this trial'):
            row=data.loc[focus]
            # Half the old width, shifted to the selected trial; keep sample counts.
            dl=float(grid['ls'][-1]-grid['ls'][0])/4
            dt=float(grid['ts'][-1]-grid['ts'][0])/4
            st.session_state.pending_search_bounds=(max(.000001,float(row['L (L☉)'])-dl),
                float(row['L (L☉)'])+dl,max(1.,float(row['Teff (K)'])-dt),float(row['Teff (K)'])+dt)
            st.session_state.search_bounds_prepared=True
            st.rerun()
        if st.session_state.pop('search_bounds_prepared',False):
            st.success('Finer bounds loaded above. Click Run grid search to evaluate them.')
        matches=data[data.Accepted].copy()
        if len(matches):
            st.subheader('Passing models')
            st.dataframe(matches[['L (L☉)','Teff (K)','r/R','M/M★','L/L★']],hide_index=True,width='stretch')
            matched=st.selectbox('Passing model to run',matches.index.tolist(),
                format_func=lambda i:f"L={data.loc[i,'L (L☉)']:.8g} L☉, Teff={data.loc[i,'Teff (K)']:.5f} K",
                key=f'passing_choice_{st.session_state.get("grid_generation",0)}')
            if st.button('Run selected passing model',type='primary'):
                row=data.loc[matched]
                result=calculate(mass,float(row['L (L☉)']),float(row['Teff (K)']),x,z)
                st.session_state.previous_trial=st.session_state.get('current')
                st.session_state.current=result
                st.session_state.pending_grid_guess=result['parameters']
                st.session_state.switch_to_model_tab=True
                st.rerun()
        else:
            st.info('No trials passed the core checks in this grid. Adjust the bounds or refine the grid and search again.')
        with st.expander('Detailed grid results table'):
            st.dataframe(data,hide_index=True,width='stretch')
        indices=list(data.index)
        choice=st.selectbox('Trial to inspect',indices,format_func=lambda i:f"{i+1}: L={data.loc[i,'L (L☉)']:.6g}, Teff={data.loc[i,'Teff (K)']:.2f} K — {data.loc[i,'Status']}")
        if st.button('Inspect selected grid trial'):
            row=data.loc[choice]
            try:
                result=calculate(mass,float(row['L (L☉)']),float(row['Teff (K)']),x,z)
                st.session_state.previous_trial=st.session_state.get('current')
                st.session_state.current=result
                st.session_state.switch_to_model_tab=True
                st.rerun()
            except Exception as exc:st.error(f'Could not inspect trial: {exc}')
        st.download_button('Download grid results (CSV)',data.to_csv(index=False),'starworks_grid.csv','text/csv')
