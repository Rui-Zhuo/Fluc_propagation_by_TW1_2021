"""Rebuild the five-case paper composite from numeric data (no raster edits)."""
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator, FormatStrFormatter
from matplotlib.lines import Line2D
from project_paths import RESULT_DIR, PROJECT_DIR


def main():
    coh=pd.read_excel(RESULT_DIR/'multiple_baseline.xlsx',sheet_name='coherency')
    info=pd.read_excel(RESULT_DIR/'multiple_baseline.xlsx',sheet_name='station_info')
    fits=pd.read_csv(RESULT_DIR/'multiple_baseline_rmse/zoom_in/summary.csv')
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':12,'axes.labelsize':13,
                         'xtick.labelsize':11,'ytick.labelsize':11,'axes.linewidth':1.1,
                         'mathtext.fontset':'dejavusans'})
    fig=plt.figure(figsize=(18,6.4))
    gs=fig.add_gridspec(2,3,width_ratios=[1,1.12,1],wspace=.17,hspace=.12,
                       left=.055,right=.985,bottom=.1,top=.98)
    axes=[fig.add_subplot(gs[0,0]),fig.add_subplot(gs[1,0]),fig.add_subplot(gs[:,1]),
          fig.add_subplot(gs[0,2]),fig.add_subplot(gs[1,2])]
    limits=[(-8.394,-8.379,3.0508,3.0568),(-8.379,-8.364,3.0508,3.0568),
            (5.3478,5.3592,1.947,1.9575),(18.578,18.589,.3678,.3722),
            (18.644,18.660,.359,.3645)]
    for i,ax in enumerate(axes):
        rows=info.iloc[4*i:4*i+3]
        pos=rows[['proj_x','proj_y']].to_numpy(float)/696000
        fit=fits.iloc[i]; v=fit.vp_km_s
        n=np.array([np.cos(np.radians(fit.theta_p_deg)),np.sin(np.radians(fit.theta_p_deg))])
        tangent=np.array([-n[1],n[0]])
        start=pos[np.argmax(abs(pos[:,0]))]
        front=start+v*n/100000
        period=(coh.iloc[4*i].scale_min+coh.iloc[4*i].scale_max)/2
        wavelength=v*period/696000
        def waves(a, inset=False):
            # Original half-plane of periodic wavefront shading.
            ts=np.linspace(-2,2,2)
            for j in range(60):
                d1=-j*wavelength/30; d2=-(j+1)*wavelength/30
                corners=np.array([front+ts[0]*tangent+d1*n,front+ts[1]*tangent+d1*n,
                                  front+ts[1]*tangent+d2*n,front+ts[0]*tangent+d2*n])
                q=(j%30)/30
                shade=.5+(abs(2*q-1)-.5)*.8
                a.fill(corners[:,0],corners[:,1],color=plt.cm.Blues(shade),lw=0)
            ends=np.array([front-2*tangent,front+2*tangent])
            a.plot(ends[:,0],ends[:,1],'r--',lw=1 if inset else 2)
        waves(ax)
        for j,k in [(0,1),(0,2),(1,2)]:
            ax.plot(pos[[j,k],0],pos[[j,k],1],'k--',lw=1.2)
        ax.scatter(pos[:,0],pos[:,1],s=18,c='k',zorder=5)
        def arrow(a,p,vec,color,lw=1.6):
            a.annotate('',xy=p+vec,xytext=p,arrowprops=dict(arrowstyle='->',color=color,lw=lw),zorder=6)
        for u,(j,k) in zip(-coh.iloc[4*i:4*i+3].vel.to_numpy(),[(0,1),(0,2),(1,2)]):
            e=pos[k]-pos[j];e/=np.linalg.norm(e)
            arrow(ax,start,u*e/100000,'orange')
        arrow(ax,start,v*n/100000,'red',2)
        radial_pos=pos[np.argmin(abs(pos[:,0]))];er=radial_pos/np.linalg.norm(radial_pos)
        arrow(ax,radial_pos,er*(.0022 if i!=2 else .002),'white',2)
        x0,x1,y0,y1=limits[i]
        label=(radial_pos-np.array([x0,y0]))/np.array([x1-x0,y1-y0])
        label=np.clip(label+np.array([0,-.10]),[.08,.12],[.87,.88])
        ax.text(*label, r'$\mathrm{e}_r$',transform=ax.transAxes,color='white',fontsize=13,
                va='center',ha='center',zorder=12,clip_on=True)
        x0,x1,y0,y1=limits[i]
        ax.set(xlim=(x0,x1),ylim=(y0,y1),xlabel=r'X ($R_\odot$)',ylabel=r'Y ($R_\odot$)')
        ax.set_aspect('equal', adjustable='box')
        ax.xaxis.set_major_locator(MaxNLocator(4));ax.yaxis.set_major_locator(MaxNLocator(4))
        ax.xaxis.set_major_formatter(FormatStrFormatter('%.3f'));ax.yaxis.set_major_formatter(FormatStrFormatter('%.3f'))
        ax.text(.94 if i<3 else .61,.84 if i!=2 else .48,rf'$v_p={v:.2f}$ km/s'+'\n'+rf'$\theta_{{kr}}={fit.theta_kr_deg:+.2f}^\circ$',
                transform=ax.transAxes,ha='right',va='top',fontsize=11.5,linespacing=1.5,zorder=15,clip_on=True,
                bbox=dict(facecolor='white',edgecolor='none',alpha=1,pad=2))
        ax.text(.96 if i<3 else .025,.96,'abcde'[i],transform=ax.transAxes,
                ha='right' if i<3 else 'left',va='top',fontsize=16,fontweight='bold')
        ins=ax.inset_axes([0,.67,.34,.33] if i<2 else ([.72,0,.28,.29] if i==2 else [.66,.67,.34,.33]))
        waves(ins,True)
        # Match data-to-display scale in X and Y so both vectors and wavefronts
        # have exactly the same apparent orientation as in the parent axes.
        ax.apply_aspect()
        parent_box=ax.get_position()
        fig.canvas.draw()
        inset_box=ins.get_position()
        ratio=(inset_box.height*fig.get_figheight())/(inset_box.width*fig.get_figwidth())
        width=inset_box.width*fig.get_figwidth()
        height=inset_box.height*fig.get_figheight()
        # Fit the entire two-wavelength shaded region plus a fixed printed
        # arrow length. The isotropic mapping preserves parent directions.
        arrow_inches=.30
        normal_extent=min(width/max(abs(n[0]),1e-9),
                          height/max(abs(n[1]),1e-9))
        inches_per_unit=(normal_extent-arrow_inches-.08)/(2*wavelength)
        arrow_length=arrow_inches/inches_per_unit
        center=front+(.5*arrow_length-wavelength)*n
        half_x=width/(2*inches_per_unit)
        half_y=height/(2*inches_per_unit)
        ins.set_xlim(center[0]-half_x,center[0]+half_x)
        ins.set_ylim(center[1]-half_y,center[1]+half_y)
        ins.annotate('',xy=front+arrow_length*n,xytext=front,
                     arrowprops=dict(arrowstyle='-|>',color='red',lw=2.6,
                                     mutation_scale=12,shrinkA=0,shrinkB=0),zorder=10)
        ins.set_xticks([]);ins.set_yticks([])
        if i==2:
            handles=[Line2D([0],[0],color='k',ls='--',label='Baseline'),Line2D([0],[0],color='r',ls='--',label='Wavefront'),Line2D([0],[0],marker='o',color='k',ls='',label='P-point')]
            ax.legend(handles=handles,loc='upper left',fontsize=11,framealpha=.9)
            arrow(ax,np.array([5.357,.0+1.9555]),np.array([.001,0]),'black')
            ax.text(.84,.84,'100 km/s',transform=ax.transAxes,ha='center',fontsize=11)
    out=PROJECT_DIR/'PAPER/Figures/2Dpropagation(6).png'
    fig.savefig(out,dpi=400,facecolor='white')
    plt.close(fig)
    print(out)

if __name__=='__main__':
    main()
