from pylab import *
import numpy as np
from matplotlib.ticker import ScalarFormatter
import matplotlib.gridspec as gridspec
from mpl_toolkits.axes_grid1.inset_locator import mark_inset
from matplotlib.ticker import AutoMinorLocator
from matplotlib.colors import LogNorm
from matplotlib.patches import Ellipse
rcParams["mathtext.fontset"]='cm'

############################### figure ###########################
#fig=figure(figsize=(15,10))     #give dimensions to the figure
##################################################################

################################ INPUT #######################################
#axes range
##############################################################################

############################ subplots ############################
#gs = gridspec.GridSpec(2,1,height_ratios=[5,2])
#ax1=plt.subplot(gs[0])
#ax2=plt.subplot(gs[1])

#make a subplot at a given position and with some given dimensions
#ax2=axes([0.4,0.55,0.25,0.1])

#gs.update(hspace=0.0,wspace=0.4,bottom=0.6,top=1.05)
#subplots_adjust(left=None, bottom=None, right=None, top=None,
#                wspace=0.5, hspace=0.5)

#set minor ticks
#ax1.xaxis.set_minor_locator(AutoMinorLocator(4))
#ax1.yaxis.set_minor_locator(AutoMinorLocator(4))


#ax1.xaxis.set_major_formatter( NullFormatter() )   #unset x label 
#ax1.yaxis.set_major_formatter( NullFormatter() )   #unset y label

#ax1.set_xticks([])
#ax1.set_yticks([])


#ax1.get_yaxis().set_label_coords(-0.2,0.5)  #align y-axis for multiple plots
##################################################################

##################### special behaviour stuff ####################
#to show error missing error bars in log scale
#ax1.set_yscale('log',nonposy='clip')  #set log scale for the y-axis

#set the x-axis in %f format instead of %e
#ax1.xaxis.set_major_formatter(ScalarFormatter()) 

#set size of ticks
#ax1.tick_params(axis='both', which='major', labelsize=10)
#ax1.tick_params(axis='both', which='minor', labelsize=8)

#set the position of the ylabel 
#ax1.yaxis.set_label_coords(-0.2, 0.4)

#set yticks in scientific notation
#ax1.ticklabel_format(axis='y',style='sci',scilimits=(1,4))

#set the x-axis in %f format instead of %e
#formatter = matplotlib.ticker.FormatStrFormatter('$%.2e$') 
#ax1.yaxis.set_major_formatter(formatter) 

#add two legends in the same plot
#ax5 = ax1.twinx()
#ax5.yaxis.set_major_formatter( NullFormatter() )   #unset y label 
#ax5.legend([p1,p2],['0.0 eV','0.3 eV'],loc=3,prop={'size':14},ncol=1)

#set points to show in the yaxis
#ax1.set_yticks([0,1,2])

#highlight a zoomed region
#mark_inset(ax1, ax2, loc1=2, loc2=4, fc="none",edgecolor='purple')
##################################################################

############################ plot type ###########################
#standard plot
#p1,=ax1.plot(x,y,linestyle='-',marker='None')

#error bar plot with the minimum and maximum values of the error bar interval
#p1=ax1.errorbar(r,xi,yerr=[delta_xi_min,delta_xi_max],lw=1,fmt='o',ms=2,
#               elinewidth=1,capsize=5,linestyle='-') 

#filled area
#p1=ax1.fill_between([x_min,x_max],[1.02,1.02],[0.98,0.98],color='k',alpha=0.2)

#hatch area
#ax1.fill([x_min,x_min,x_max,x_max],[y_min,3.0,3.0,y_min],#color='k',
#         hatch='X',fill=False,alpha=0.5)

#scatter plot
#p1=ax1.scatter(k1,Pk1,c='b',edgecolor='none',s=8,marker='*')

#plot with markers
#pl4,=ax1.plot(ke3,Pk3/Pke3,marker='.',markevery=2,c='r',linestyle='None')

#set size of dashed lines
#ax.plot([0, 1], [0, 1], linestyle='--', dashes=(5, 1)) #length of 5, space of 1

#image plot
#cax = ax1.imshow(densities,cmap=get_cmap('jet'),origin='lower',
#           extent=[x_min, x_max, y_min, y_max],
#           #vmin=min_density,vmax=max_density)
#           norm = LogNorm(vmin=min_density,vmax=max_density))
#cbar = fig.colorbar(cax, ax2, ax=ax1, ticks=[-1, 0, 1]) #in ax2 colorbar of ax1
#cbar.set_label(r"$M_{\rm CSF}\/[h^{-1}M_\odot]$",fontsize=14,labelpad=-50)
#cbar.ax.tick_params(labelsize=10)  #to change size of ticks

#make a polygon
#polygon = Rectangle((0.4,50.0), 20.0, 20.0, edgecolor='purple',lw=0.5,
#                    fill=False)
#ax1.add_artist(polygon)
####################################################################

x_min, x_max = 2.5e-2, 1.1
y_min, y_max = 7e-4, 12

f_out='Toy_model.pdf'

f1 = '../results/fit_errors/errors_Pk-30-30-30-2_kpivot=2.00.txt'
f2 = '../results/fit_errors/errors_Pk-30-30-30-2_kpivot=0.50_varied_A.txt'
f3 = '../results/fit_errors/errors_Pk-30-30-30-2_kpivot=0.50.txt'
k1, dA_NN1, dB_NN1, dA_LS, dB_LS = np.loadtxt(f1,unpack=True)
k2, dA_NN2, dB_NN2                 = np.loadtxt(f2,unpack=True)
k3, dA_NN3, dB_NN3                 = np.loadtxt(f3,unpack=True) 

fig=figure(figsize=(15,6))
ax1=fig.add_subplot(121)
ax2=fig.add_subplot(122) 

for ax in [ax1,ax2]:
    ax.set_xscale('log')
    ax.set_yscale('log')
    
    ax.set_xlim([x_min,x_max])
    ax.set_ylim([y_min,y_max])

    ax.set_xlabel(r'$k_{\rm max}\/[h\/{\rm Mpc}^{-1}]$',fontsize=18)
    ax.set_ylabel(r'${\rm error}$',fontsize=22)

p1,=ax1.plot(k1,dA_NN1,linestyle='-',marker='None', c='r')
p2,=ax1.plot(k1,dB_NN1,linestyle='-',marker='None', c='b')
p3,=ax1.plot(k1,dA_LS, linestyle='--',marker='None', c='r')
p4,=ax1.plot(k1,dB_LS, linestyle='--',marker='None', c='b')

p5,= ax2.plot(k1,dA_NN1,linestyle='-',marker='None', c='r')
p6,= ax2.plot(k1,dB_NN1,linestyle='-',marker='None', c='b')
p7,= ax2.plot(k2,dA_NN2,linestyle='-.',marker='None', c='r')
p8,= ax2.plot(k2,dB_NN2,linestyle='-.',marker='None', c='b')
p9,= ax2.plot(k3,dA_NN3,linestyle=':',marker='None', c='r')
p10,=ax2.plot(k3,dB_NN3,linestyle=':',marker='None', c='b')


#place a label in the plot
ax1.text(3e-2,2e-3, 'AstroNo: no baryon effects', fontsize=17, color='k')
ax2.text(3e-2,2e-3, 'neural networks', fontsize=17, color='k')

#legend
ax1.legend([p1,p3,p2,p4],
           [r"$A:\,\,{\rm neural\,\,network}$",
            r"$A:\,\,{\rm max\,\,likelihood}$",
            r"$B:\,\,{\rm neural\,\,network}$",
            r"$B:\,\,{\rm max\,\,likelihood}$"],
           loc=0,prop={'size':15},ncol=1,frameon=True)


ax2.legend([p5,p7,p9,p6,p8,p10],
           [r"$A:\,\,{\rm AstroNo}$",
            r"$A:\,\,{\rm AstroDis}$",
            r"$A:\,\,{\rm AstroCon}$",
            r"$B:\,\,{\rm AstroNo}$",
            r"$B:\,\,{\rm AstroDis}$",
            r"$B:\,\,{\rm AstroCon}$"],
           loc=0,prop={'size':15},ncol=2,frameon=True)


#ax1.set_title(r'$\sum m_\nu=0.0\/{\rm eV}$',position=(0.5,1.02),size=18)
#title('About as simple as it gets, folks')
#suptitle('About as simple as it gets, folks')  #for title with several panels
#grid(True)
#show()
savefig(f_out, bbox_inches='tight')
close(fig)










###############################################################################
#some useful colors:

#'darkseagreen'
#'yellow'
#"hotpink"
#"gold
#"fuchsia"
#"lime"
#"brown"
#"silver"
#"cyan"
#"dodgerblue"
#"darkviolet"
#"magenta"
#"deepskyblue"
#"orchid"
#"aqua"
#"darkorange"
#"coral"
#"lightgreen"
#"salmon"
#"bisque"
