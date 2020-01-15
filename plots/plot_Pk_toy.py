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

x_min, x_max = 1e-2, 1.1
y_min, y_max = 7e-4, 12

kmin = 7e-3
kmax = 2.0

f_out = 'Toy_model_Pk.pdf'

fig = figure()
ax1 = fig.add_subplot(111)

ax1.set_xscale('log')
ax1.set_yscale('log')
    
ax1.set_xlim([x_min,x_max])
#ax1.set_ylim([y_min,y_max])

ax1.set_xlabel(r'$k_{\rm max}\/[h\/{\rm Mpc}^{-1}]$',fontsize=18)
ax1.set_ylabel(r'$P(k)\,[(h^{-1}{\rm Mpc})^3]$',fontsize=22)




seed = 5


np.random.seed(seed)

# find the fundamental frequency, the number of bins up to kmax and the k-array
kF     = kmin
k_bins = int((kmax-kmin)/kF)
k      = np.arange(2,k_bins+2)*kF #avoid k=kF as we will get some negative values
Nk     = 4.0*np.pi*k**2*kF/kF**3  #number of modes in each k-bin


# model 0 
A = 5.0
B = -0.6
Pk = A*k**B
dPk = np.sqrt(Pk**2/Nk)
Pk  = np.random.normal(loc=Pk, scale=dPk)
p1,=ax1.plot(k,Pk,linestyle='-',marker='None', c='r')




# model 1
A = 6.0
B = -0.8
D = 0.5
kpivot = 0.5
fix_A_value = True

# get the hydro Pk part
Pk = A*k**B
indexes = np.where(k>kpivot)[0]
if len(indexes)>0:
    A_value = Pk[indexes[0]]/k[indexes[0]]**D
    if not(fix_A_value):
        A_value = A_value*(0.8 + np.random.random()*0.4)
Pk[indexes] = A_value*k[indexes]**D
dPk = np.sqrt(Pk**2/Nk)
Pk  = np.random.normal(loc=Pk, scale=dPk)
p2,=ax1.plot(k,Pk,linestyle='-',marker='None', c='b')



# model 2
A = 5.0
B = -0.1
D = -0.5
kpivot = 0.5
fix_A_value = False

# get the hydro Pk part
Pk = A*k**B
indexes  = np.where(k>kpivot)[0]
indexes2 = np.where(k<kpivot)[0]
if len(indexes)>0:
    A_value = Pk[indexes[0]]/k[indexes[0]]**D
    if not(fix_A_value):
        A_value = A_value*(0.8 + np.random.random()*0.4)
Pk[indexes] = A_value*k[indexes]**D
dPk = np.sqrt(Pk**2/Nk)
Pk  = np.random.normal(loc=Pk, scale=dPk)
p3,=ax1.plot(k[indexes2],Pk[indexes2],linestyle='-',marker='None', c='g')
p3,=ax1.plot(k[indexes],Pk[indexes],linestyle='-',marker='None', c='g')




#legend
ax1.legend([p1,p3,p2],
           [r"${\rm model\,\,0}$",
            r"${\rm model\,\,1}$",
            r"${\rm model\,\,2}$"],
           loc=0,prop={'size':15},ncol=1,frameon=True)



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
