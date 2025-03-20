import matplotlib.colors as mcolors
import numpy as np
import matplotlib.dates as mdates

def Fire_and_Ice():

    # First define 'Blue1' from ColorMoves (used for negative vorticiy values)
    x=[0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,0.95,1]
    r=[0.011765,0.066667,0.117647,0.164706,0.235294,0.290196,0.356863,0.439216,0.521569,0.643137,0.780392,0.929412]
    g=[0.015686,0.078431,0.168627,0.235294,0.333333,0.415686,0.501961,0.611765,0.717647,0.831373,0.92549,1.]
    b=[0.101961,0.2,0.321569,0.4,0.501961,0.580392,0.65098,0.729412,0.8,0.878431,0.94902,1.]
    o=np.zeros(256)+1

    xi=np.linspace(0,1,256)
    ri=np.interp(xi,x,r)
    gi=np.interp(xi,x,g)
    bi=np.interp(xi,x,b)

    colors=np.reshape([ri,gi,bi,o],[4,256])
    colors=colors.T
    Blue1 = mcolors.LinearSegmentedColormap.from_list('my_colormap', np.flip(colors,axis=0))


    # Second, define 'ExpHue7' from ColorMoves (used for positive vorticiy values)
    x=np.linspace(0,1,11)
    r=[1.,0.980392,0.94902,0.909804,0.85098,0.780392,0.670588,0.501961,0.34902,0.239216,0.14902]
    g=[0.976471,0.917647,0.831373,0.713725,0.533333,0.376471,0.25098,0.145098,0.07451,0.031373,0.003922]
    b=[0.780392,0.509804,0.360784,0.254902,0.168627,0.109804,0.07451,0.043137,0.027451,0.011765,0.]

    ri=np.interp(xi,x,r)
    gi=np.interp(xi,x,g)
    bi=np.interp(xi,x,b)

    colors=np.reshape([ri,gi,bi,o],[4,256])
    colors=colors.T
    ExpHue7 = mcolors.LinearSegmentedColormap.from_list('my_colormap', np.flip(colors,axis=0))


    # sample the colormaps that you want to use. Use 128 from each so we get 256
    # colors in total
    colors_low = Blue1(np.linspace(0., 1, 128))
    colors_high = ExpHue7(np.linspace(0, 1, 128))

    # combine them and build a new colormap
    colors = np.vstack((colors_low, colors_high))
    Fire_n_Ice = mcolors.LinearSegmentedColormap.from_list('my_colormap', colors)
    
    return Fire_n_Ice

def generate_nice_ticks(data, percentiles, diverging=False, fix_zero = False, num_ticks=5):
    
    # Calculate the range of the data
    if diverging == False:
        data_min, data_max = np.nanpercentile(data,percentiles)
        if fix_zero == True:
            data_min = 0
        data_range = data_max - data_min
    if diverging == True:
        data_min, data_max = np.nanpercentile(np.abs(data),percentiles)
        data_range = 2*data_max
    
    # Estimate a preliminary step size
    raw_step = data_range / (num_ticks-1)
    
    # Calculate the magnitude of the step size
    magnitude = np.floor(np.log10(raw_step))
    
    # Refine step size to a nice figure
    refined_step = round(raw_step / 10**magnitude) * 10**magnitude
    
    # Find nice lower and upper bounds
    if diverging == False:
        nice_min = np.ceil(data_min / refined_step) * refined_step
        nice_max = np.floor(data_max / refined_step) * refined_step
    if diverging == True:
        nice_min = -np.floor(data_max / refined_step) * refined_step
        nice_max = np.floor(data_max / refined_step) * refined_step
    
    # Generate tick values
    if diverging == False:
        ticks = np.arange(nice_min, nice_max + 0.5 * refined_step, refined_step)
    if diverging == True:
        ticks = np.arange(nice_min, nice_max + refined_step, refined_step)

    return ticks

def cmap_range(data, percentiles, diverging=False, fix_zero=False, num_levels=31):
    
    # Calculate the range of the data
    if diverging == False:
        data_min, data_max = np.nanpercentile(data,percentiles)
        if fix_zero == True:
            data_min = 0
        data_range= data_max - data_min
    if diverging == True:
        data_min, data_max = np.nanpercentile(np.abs(data),percentiles)
        data_range= data_max - data_min
        if data_range == 0:
            data_min, data_max = np.nanpercentile(np.abs(data),[0,100])
            data_range= 2*data_max
    
    if data_range > 0:
        # Estimate a preliminary step size
        raw_step = data_range / (num_levels - 1)

        # Calculate the magnitude of the step size
        magnitude = np.floor(np.log10(raw_step))

        # Refine step size to a nice figure
        refined_step = round(raw_step / 10**magnitude) * 10**magnitude

        # Find nice lower and upper bounds
        if diverging == False:
            nice_min = np.floor(data_min / refined_step) * refined_step
            nice_max = np.ceil(data_max / refined_step) * refined_step
        if diverging == True:
            nice_min = -np.floor(data_max / refined_step) * refined_step
            nice_max = np.floor(data_max / refined_step) * refined_step
        levels = np.linspace(nice_min,nice_max,num_levels)
    
    if data_range == 0:
        levels = 0
    
    return levels

def timelab_format_func(value, tick_number):
    dt = mdates.num2date(value)
    if dt.hour == 0:
        return "{}\n{}".format(dt.strftime("%Y-%b-%d"), dt.strftime("%H"))
    else:
        return dt.strftime("%H")