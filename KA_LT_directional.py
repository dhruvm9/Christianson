
###############################################################################
## Linear Track
#to do: id PCs on linear track, i have to isolate dir before defining cells as HD or non-HD
###############################################################################

position_=remove_immobility(position,speed_threshold = 0.06)


pos= position.restrict(ep2)[['x','z']]
clean_frames = pos.z.values < 0.2

pos_tmp = pos [clean_frames]


pos_x = gaussian_filter(pos_tmp.x.values, sigma=100)
pos_y = gaussian_filter(pos_tmp.z.values, sigma=100)


figure();plot(pos_tmp.x,pos_tmp.z)



figure();plot(pos_x)

# needs fine-tuning to generalize across animals
# if stationary ts are removed from the original position data, the trajectories will be more seperable
peaks,_= scipy.signal.find_peaks(pos_x, height = 0.3,distance=4000)
troughs,_ = scipy.signal.find_peaks((pos_x)*-1, height= 0.26,distance=2500)

peaks_ts = pos_tmp.index[peaks]
troughs_ts = pos_tmp.index[troughs] 

# rightwards
r_run_ep = nts.IntervalSet(start=troughs_ts, end=peaks_ts)


# leftwards
l_run_ep = nts.IntervalSet(start=peaks_ts, end=troughs_ts[1:])



###############################################################################
## Plots by direction
###############################################################################
rights = pos_tmp.restrict(r_run_ep)[['x','z']]
lefts = pos_tmp.restrict(l_run_ep)[['x','z']]


eps = l_run_ep
figure()
for i in range(10):
    subplot(10,1,i+1)
    ep_tmp = nts.IntervalSet(start=eps.loc[i].start,end=eps.loc[i].end ) 
    xpos = pos_tmp.restrict(ep_tmp)['x']
    ypos = pos_tmp.restrict(ep_tmp)['z']
    plot(xpos,ypos)
