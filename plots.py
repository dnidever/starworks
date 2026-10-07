"""Compact profiles with explicit invalid-value and core-boundary markers."""
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

def invalid_shells(data):
    bad=~np.isfinite(data[['r','M','L','T','P','rho','epsilon','kappa']]).all(axis=1)
    for q in ['T','P','rho']: bad |= data[q]<=0
    for q in ['r','M','L','epsilon','kappa']: bad |= data[q]<0
    # Only the deliberately zero-valued starting surface shell is exempt.
    surface=(data['index']==1)&(data['T']==0)&(data.P==0)&(data.rho==0)&(data.r==data.r.max())
    bad &= ~surface
    return np.asarray(bad)

def make_profiles(current, reference, coord, axis, log=False, advanced=False, log_x=False, x_range=None, show_core=False, show_points=False):
    panels=[[('P','Pressure','dyn cm⁻²')],[('kappa','Opacity','cm² g⁻¹')],[('dlnPdlnT','d ln P / d ln T','')],[('transport','Transport','')]] if advanced else [
        [('T','Temperature','K'),('rho','Density','g cm⁻³')],
        [('m_fraction','Mass fraction',''),('l_fraction','Luminosity fraction','')],
        [('epsilon','Energy generation','erg g⁻¹ s⁻¹')],
        [('rho','Density','g cm⁻³')]]
    titles=['Pressure','Opacity','Temperature gradient','Energy transport'] if advanced else ['Temperature and density / respective maxima','Enclosed mass and luminosity / totals','Nuclear energy generation','Density (g cm⁻³)']
    fig=make_subplots(rows=2,cols=2,subplot_titles=titles)
    for i,panel in enumerate(panels):
        row,col=i//2+1,i%2+1
        for model,name,dash in [(reference,'Comparison','solid'),(current,'Current','solid')]:
            if model is None: continue
            data=model['profile']
            for j,(q,label,unit) in enumerate(panel):
                raw=(data.zone=='c').astype(float).to_numpy() if q=='transport' else data[q].to_numpy()
                y=raw.copy()
                if not advanced and i==0 and q in ['T','rho']:
                    pos=raw[np.isfinite(raw)&(raw>0)]
                    y=raw/pos.max() if len(pos) else np.full(len(raw),np.nan)
                if log and (advanced and q in ['P','kappa'] or not advanced and i in [0,2,3]):
                    y=np.where(y>0,y,np.nan)
                    fig.update_yaxes(type='log',row=row,col=col)
                xvalues=data[coord].to_numpy().copy()
                if log_x:
                    valid_x=np.isfinite(xvalues)&(xvalues>0)
                    xvalues=np.where(valid_x,xvalues,np.nan)
                    y=np.where(valid_x,y,np.nan)
                fig.add_trace(go.Scatter(x=xvalues,y=y,customdata=raw,mode='lines+markers' if show_points else 'lines',opacity=.35 if name=='Comparison' else 1.0,marker=dict(size=4 if name=='Comparison' else 5,symbol='circle-open' if name=='Comparison' else 'circle'),name=f'{name}: {label}' if reference else label,legend='legend' if i==0 else f'legend{i+1}',line=dict(color=(['#f59e0b','#38bdf8'] if name=='Comparison' else ['#b45309','#0369a1'])[j],width=7 if name=='Comparison' else 2,dash=dash,shape='hv' if q=='transport' else 'linear'),hovertemplate=f'{label}: %{{customdata:.4g}} {unit}<br>{axis}: %{{x:.4g}}<extra>{name}</extra>'),row=row,col=col)
                if show_core and q!='transport':
                    core=model['core']
                    raw_core=core['M']/(model['parameters'][0]*1.989e33) if q=='m_fraction' else core['L']/(model['parameters'][1]*3.826e33) if q=='l_fraction' else core[q]
                    core_x=0.0  # At the geometric center, enclosed mass coordinate is zero.
                    core_y=raw_core
                    if not advanced and i==0 and q in ['T','rho']:
                        core_y=raw_core/pos.max() if len(pos) else np.nan
                    logarithmic_y=log and (advanced and q in ['P','kappa'] or not advanced and i in [0,2,3])
                    if np.isfinite(core_x) and np.isfinite(core_y) and (not log_x or core_x>0) and (not logarithmic_y or core_y>0):
                        if q in ['m_fraction','l_fraction']:
                            valid_end=np.flatnonzero(np.isfinite(xvalues)&np.isfinite(y))
                            if len(valid_end):
                                # A dotted segment distinguishes the extrapolation from integrated shells.
                                endpoint=valid_end[0]
                                fig.add_trace(go.Scatter(x=[xvalues[endpoint],core_x],y=[y[endpoint],core_y],
                                    mode='lines',showlegend=False,name=f'{name}: core connection',
                                    opacity=.35 if name=='Comparison' else 1.0,
                                    line=dict(color=(['#f59e0b','#38bdf8'] if name=='Comparison' else ['#b45309','#0369a1'])[j],width=7 if name=='Comparison' else 2,dash='dot'),
                                    hovertemplate='Connection to extrapolated core<extra>'+name+'</extra>'),row=row,col=col)
                        fig.add_trace(go.Scatter(x=[core_x],y=[core_y],mode='markers',showlegend=False,
                            opacity=.35 if name=='Comparison' else 1.0,
                            marker=dict(symbol='diamond-open',size=15 if name=='Comparison' else 12,color='red' if name=='Current' and q=='rho' and current['flag']==1 else (['#f59e0b','#38bdf8'] if name=='Comparison' else ['#b45309','#0369a1'])[j],line=dict(width=2)),
                            name=f'{name}: extrapolated core',customdata=[raw_core],
                            hovertemplate=f'Extrapolated core: {label}=%{{customdata:.4g}} {unit}<br>{axis}=%{{x:.4g}}<extra>{name}</extra>'),row=row,col=col)
        if not advanced and i==1:
            # Identify actual offending shell values; never invent a negative
            # residual for a core mismatch that only concerns density or temperature.
            data=current['profile']
            for q,label in [('m_fraction','Mass'),('l_fraction','Luminosity')]:
                failed=data[(data[q]<0)&np.isfinite(data[q])&np.isfinite(data[coord])]
                if log_x: failed=failed[failed[coord]>0]
                if len(failed):
                    point=failed.iloc[0]
                    fig.add_trace(go.Scatter(x=[point[coord]],y=[point[q]],mode='markers',
                        name=f'{label} became negative',legend='legend2',
                        marker=dict(size=14,color='red',symbol='circle',line=dict(color='white',width=1)),
                        hovertemplate=f'{label} became negative<br>{axis}=%{{x:.5g}}<br>Fraction=%{{y:.5g}}<extra>Invalid shell</extra>'),row=row,col=col)
        # Radius bands are only meaningful with the radius axis; invalid mass coordinates
        # can fold back on themselves, so use shell markers on that axis.
        data=current['profile']; bad=invalid_shells(data); xx=data[coord].to_numpy()
        if coord=='r_fraction':
            for k in np.flatnonzero(bad):
                left=(xx[k-1]+xx[k])/2 if k else xx[k]
                right=(xx[k]+xx[k+1])/2 if k+1<len(xx) else xx[k]
                if np.isfinite(left) and np.isfinite(right) and (not log_x or left>0 and right>0):
                    fig.add_vrect(x0=left,x1=right,fillcolor='red',opacity=.16,line_width=0,row=row,col=col)
        for k in np.flatnonzero(bad):
            if np.isfinite(xx[k]) and (not log_x or xx[k]>0): fig.add_vline(x=float(xx[k]),line_color='red',line_width=1,row=row,col=col)
        if current['flag']!=0 or current['error']:
            # Shade the unresolved interior from the innermost valid integrated shell.
            good=data[(data.r>0)&~bad&np.isfinite(data[coord])]
            if coord=='r_fraction' and len(good):
                boundary=float(good.iloc[0][coord])
                if log_x:
                    # Zero is outside a log axis: shade to the visible positive lower edge.
                    candidates=data.loc[(data[coord]>0)&np.isfinite(data[coord]),coord]
                    lower=float(x_range[0]) if x_range else float(candidates.min())*.8
                else:
                    lower=0.0
                if lower<boundary:
                    fig.add_vrect(x0=lower,x1=boundary,fillcolor='red',opacity=.16,line_width=0,row=row,col=col)
            positive=data[(data.r>0)&np.isfinite(data[coord]) & ((data[coord]>0) if log_x else True)]
            if len(positive):
                fig.add_vline(x=float(positive.iloc[0][coord]),line_color='red',line_dash='dash',line_width=2,row=row,col=col)
        fig.update_xaxes(title_text=axis,type='log' if log_x else 'linear',row=row,col=col)
        if advanced and i==3:
            fig.update_yaxes(tickvals=[0,1],ticktext=['Radiative','Convective'],range=[-.1,1.1],row=row,col=col)
        if not advanced and i==3: fig.update_yaxes(title_text='g cm⁻³',row=row,col=col)
        if not advanced and i==2: fig.update_yaxes(title_text='erg g⁻¹ s⁻¹',row=row,col=col)
    if not advanced and current['flag']==1 and current['error']==0:
        data=current['profile']
        shells=data[(data.r>0)&np.isfinite(data.rho)&(data.rho>0)]
        if len(shells)>=2:
            rho_inner=float(shells.iloc[0].rho)
            rho_max=10*rho_inner*rho_inner/float(shells.iloc[1].rho)
            rho_core=current['core']['rho']
            relation='below last shell' if rho_core<rho_inner else 'above upper limit'
            fig.add_annotation(x=fig.layout.xaxis4.domain[0]+.015,
                y=fig.layout.yaxis4.domain[0]+.3*(fig.layout.yaxis4.domain[1]-fig.layout.yaxis4.domain[0]),xref='paper',yref='paper',
                text=f'Core density mismatch: {rho_core:.4g} g/cm³ ({relation})<br>Allowed: {rho_inner:.4g} to {rho_max:.4g} g/cm³',
                showarrow=False,xanchor='left',yanchor='top',font=dict(color='#b91c1c',size=11),
                bgcolor='rgba(255,240,240,0.9)')
    for i in range(len(panels)):
        suffix='' if i==0 else str(i+1)
        xdomain=fig.layout['xaxis'+suffix].domain
        ydomain=fig.layout['yaxis'+suffix].domain
        bottom=not advanced and i in [0,1,2]
        left=not advanced and i in [0,2,3]
        fig.update_layout(**{'legend'+suffix:dict(
            x=xdomain[0]+.2*(xdomain[1]-xdomain[0]) if left else xdomain[1]-.01,
            y=ydomain[0]+.015 if bottom else ydomain[1]-.015,
            xanchor='left' if left else 'right',yanchor='bottom' if bottom else 'top',
            orientation='v',font=dict(size=10),bgcolor='rgba(255,255,255,0.85)',
            bordercolor='rgba(100,100,100,0.3)',borderwidth=1)})
    fig.update_layout(height=720 if not advanced else 600,margin=dict(t=60,b=45),hovermode='x unified')
    if x_range is not None:
        bounds=np.log10(x_range).tolist() if log_x else list(x_range)
        fig.update_xaxes(range=bounds,autorange=False)
    elif coord=='r_fraction' and not log_x:
        xmax=max(float(model['profile'].r_fraction.max()) for model in [current,reference] if model is not None)
        fig.update_xaxes(range=[0,xmax],autorange=False)
    return fig
