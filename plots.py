"""Compact profiles with explicit invalid-value and core-boundary markers."""
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

def invalid_shells(data):
    bad=~np.isfinite(data[['r','M','L','T','P','rho','epsilon','kappa']]).all(axis=1)
    for q in ['T','P','rho']: bad |= data[q]<=0
    for q in ['r','M','L','epsilon','kappa']: bad |= data[q]<0
    return np.asarray(bad)

def make_profiles(current, reference, coord, axis, log=False, advanced=False):
    panels=[[('P','Pressure','dyn cm⁻²')],[('kappa','Opacity','cm² g⁻¹')],[('dlnPdlnT','d ln P / d ln T','')]] if advanced else [
        [('T','Temperature','K'),('rho','Density','g cm⁻³')],
        [('m_fraction','Mass fraction',''),('l_fraction','Luminosity fraction','')],
        [('epsilon','Energy generation','erg g⁻¹ s⁻¹')],
        [('transport','Transport','')]]
    titles=['Pressure','Opacity','Temperature gradient'] if advanced else ['Temperature and density / respective maxima','Enclosed mass and luminosity / totals','Nuclear energy generation','Energy transport']
    fig=make_subplots(rows=2,cols=2,subplot_titles=titles)
    for i,panel in enumerate(panels):
        row,col=i//2+1,i%2+1
        for model,name,dash in [(current,'Current','solid'),(reference,'Comparison','dash')]:
            if model is None: continue
            data=model['profile']
            for j,(q,label,unit) in enumerate(panel):
                raw=(data.zone=='c').astype(float).to_numpy() if q=='transport' else data[q].to_numpy()
                y=raw.copy()
                if not advanced and q in ['T','rho']:
                    pos=raw[np.isfinite(raw)&(raw>0)]
                    y=raw/pos.max() if len(pos) else np.full(len(raw),np.nan)
                if log and (advanced and q in ['P','kappa'] or not advanced and i in [0,2]):
                    y=np.where(y>0,y,np.nan)
                    fig.update_yaxes(type='log',row=row,col=col)
                fig.add_trace(go.Scatter(x=data[coord],y=y,customdata=raw,mode='lines',name=f'{name}: {label}' if reference else label,legend='legend' if i==0 else f'legend{i+1}',line=dict(color=['#f59e0b','#38bdf8'][j],dash=dash,shape='hv' if q=='transport' else 'linear'),hovertemplate=f'{label}: %{{customdata:.4g}} {unit}<br>{axis}: %{{x:.4g}}<extra>{name}</extra>'),row=row,col=col)
        # Radius bands are only meaningful with the radius axis; invalid mass coordinates
        # can fold back on themselves, so use shell markers on that axis.
        data=current['profile']; bad=invalid_shells(data); xx=data[coord].to_numpy()
        if coord=='r_fraction':
            for k in np.flatnonzero(bad):
                left=(xx[k-1]+xx[k])/2 if k else xx[k]
                right=(xx[k]+xx[k+1])/2 if k+1<len(xx) else xx[k]
                if np.isfinite(left) and np.isfinite(right):
                    fig.add_vrect(x0=left,x1=right,fillcolor='red',opacity=.16,line_width=0,row=row,col=col)
        for k in np.flatnonzero(bad):
            if np.isfinite(xx[k]): fig.add_vline(x=float(xx[k]),line_color='red',line_width=1,row=row,col=col)
        if current['flag']!=0 or current['error']:
            positive=data[(data.r>0)&np.isfinite(data[coord])]
            if len(positive):
                fig.add_vline(x=float(positive.iloc[0][coord]),line_color='red',line_dash='dash',line_width=2,row=row,col=col)
        fig.update_xaxes(title_text=axis,row=row,col=col)
        if not advanced and i==3:
            fig.update_yaxes(tickvals=[0,1],ticktext=['Radiative','Convective'],range=[-.1,1.1],row=row,col=col)
        if not advanced and i==2: fig.update_yaxes(title_text='erg g⁻¹ s⁻¹',row=row,col=col)
    for i in range(len(panels)):
        suffix='' if i==0 else str(i+1)
        xdomain=fig.layout['xaxis'+suffix].domain
        ydomain=fig.layout['yaxis'+suffix].domain
        fig.update_layout(**{'legend'+suffix:dict(
            x=xdomain[1]-.01,y=ydomain[1]-.015,xanchor='right',yanchor='top',
            orientation='v',font=dict(size=10),bgcolor='rgba(255,255,255,0.85)',
            bordercolor='rgba(100,100,100,0.3)',borderwidth=1)})
    fig.update_layout(height=720 if not advanced else 600,margin=dict(t=60,b=45),hovermode='x unified')
    return fig
