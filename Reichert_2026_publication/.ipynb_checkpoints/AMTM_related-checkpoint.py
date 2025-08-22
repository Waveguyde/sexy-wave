import xarray as xr
import numpy as np
from scipy import stats
from tqdm import tqdm
from scipy.optimize import curve_fit
from scipy.ndimage import gaussian_filter

def gradient_filtering(data,sMAD_factor=3):

    print('Start: Gradient Filtering')
    
    dt_dx = np.gradient(data,axis=0)
    dt_dy = np.gradient(data,axis=1)
    dt_dt = np.gradient(data,axis=2)
    
    sMAD_x=1.4826*stats.median_abs_deviation(dt_dx,axis=None,nan_policy='omit')
    sMAD_y=1.4826*stats.median_abs_deviation(dt_dy,axis=None,nan_policy='omit')
    sMAD_t=1.4826*stats.median_abs_deviation(dt_dt,axis=None,nan_policy='omit')
    
    gradient_filter = (
        (dt_dx < - sMAD_factor * sMAD_x) |
        (dt_dx >   sMAD_factor * sMAD_x) |
        (dt_dy < - sMAD_factor * sMAD_y) |
        (dt_dy >   sMAD_factor * sMAD_y) |
        (dt_dt < - sMAD_factor * sMAD_t) |
        (dt_dt >   sMAD_factor * sMAD_t)
    )

    print('Finish: Gradient Filtering')
    
    return np.where(~gradient_filter,data,np.nan)

def get_mean_sMAD(data):
    
    FOV_sMAD=[]
    for t in range(data.shape[2]):
        sMAD=1.4826*stats.median_abs_deviation(data[:,:,t],axis=None,nan_policy='omit')
        FOV_sMAD=np.append(FOV_sMAD,sMAD)
    
    return np.mean(FOV_sMAD)

def sMAD_filtering(data,filter_threshold=5):

    print('Start: sMAD Filtering')
    data_filt = np.empty(data.shape)
    mean_sMAD = get_mean_sMAD(data)

    for t in tqdm(range(data.shape[2])):
        median=np.nanmedian(data[:,:,t])

        lower_threshold, upper_threshold = median - filter_threshold * mean_sMAD, median + filter_threshold * mean_sMAD
        sMAD_filter = (
            (data[:,:,t] < lower_threshold) |
            (data[:,:,t] > upper_threshold)
        )
        data_filt[:,:,t] = np.where(~sMAD_filter,data[:,:,t],np.nan)

    print('Finish: sMAD Filtering')
    return data_filt
        
def preprocessing_AMTM(data_path,dx=625,dy=625,dt=35):
    """
    Set unphysical large temperature gradients in x,y,time to Nan.
    Set Outliers to Nan. These are mostly edge effects due to street lamps, the moon, or clouds.

    Parameters
    ----------
    data_path : str
        Path where AMTM data is stored as netCDF.
        
    dx, dy, dt : floats
        Resolution in x,y,time (meter,meter,second).

    Returns
    -------
    xr.DataSet
        The "cleaned" AMTM temperature data set.
    """
    
    ds    = xr.open_dataset(data_path,engine='netcdf4')
    # Cut the data due to edge effects. This is hard coded for the AMTM data obtained in Rio Grande, Argentina, but might need other settings for different location.
    temp  = ds['temperature'].isel(x=slice(0,-31),y=slice(0,-8),time=slice(1,-2)).values 
    temp  = np.flip(temp,axis=0)

    x     = np.linspace(ds['y'].isel(y=0).values,(ds.coords['y'].size-1)*dx,ds.coords['y'].size)
    y     = np.linspace(ds['x'].isel(x=0).values,(ds.coords['x'].size-1)*dy,ds.coords['x'].size)

    temp_filt     = gradient_filtering(temp) 
    temp_filtfilt = sMAD_filtering(temp_filt)

    return temp_filtfilt

def tides(t, A24, phi24, A12, phi12, A8, phi8):
    return (A24 * np.sin(2 * np.pi / (24*3600) * t + phi24) +
            A12 * np.sin(2 * np.pi / (12*3600) * t + phi12) +
            A8 * np.sin(2 * np.pi / (8*3600) * t + phi8))

def get_basis(x, y, max_order=1):
    #Return the fit basis polynomials: 1, x, x^2, ..., xy, x^2y, ... etc.
    basis = []
    for i in range(max_order+1):
        for j in range(max_order - i +1):
            basis.append(x**j * y**i)
    return basis

def calculate_2dft(input):
    ft = np.fft.ifftshift(input)
    ft = np.fft.fft2(ft)
    return np.fft.fftshift(ft)

def calculate_2dift(input):
    ift = np.fft.ifftshift(input)
    ift = np.fft.ifft2(ift)
    ift = np.fft.fftshift(ift)
    return ift.real

def AMTM_filter(data):
    
    data-=np.nanmean(data)
    
    ny, nx = data.shape
    y0 = np.arange(ny)#linspace(0, (len(signal[:, 0]) - 1) * 0.625, ny)
    x0 = np.arange(nx)#linspace(0, (len(signal[0, :]) - 1) * 0.625, nx)
    dx, dy = 1, 1#(x0[1] - x0[0]), (y0[1] - y0[0])

    X, Y = np.meshgrid(x0,y0)
    x, y = X.flatten(), Y.flatten()
    b = data.ravel()
    mask = ~np.isnan(b)
    b = b[mask]
    x = x[mask]
    y = y[mask]
    
    max_order = 1
    basis = get_basis(x, y, max_order)

    A = np.vstack(basis).T
    c, r, rank, s = np.linalg.lstsq(A, b, rcond=None)

    # Calculate the fitted surface from the coefficients, c.
    fit = np.sum(c[:, None, None] * np.array(get_basis(X, Y, max_order)).reshape(len(basis), *X.shape), axis=0)
    
    detrended_data=data-fit
    detrended_data[np.isnan(detrended_data)]=0

    ft = calculate_2dft(detrended_data)
    freqs_x    = np.fft.fftfreq(nx, 1)
    freqs_x    = np.fft.fftshift(freqs_x)
    freqs_y    = np.fft.fftfreq(ny, 1)
    freqs_y    = np.fft.fftshift(freqs_y)
    freqs_X, freqs_Y = np.meshgrid(freqs_x,freqs_y)

    filtered_ft=ft.copy()
    filtered_ft[int(ny/2)-1:int(ny/2)+2,int(nx/2)-1:int(nx/2)+2]=0
    highpass_data=calculate_2dift(filtered_ft)
    lowpass_data=detrended_data-highpass_data
    
    return highpass_data, fit+lowpass_data 

def AMTM_decomposition(da,dt=35,gauss_window_std=3,init_tides_guess = [1.0, 0.0, 10.0, 0.0, 8.0, 0.0]):

    time         = np.linspace(0,(da.coords['time'].size-1)*dt,da.coords['time'].size)
    total_mean   = da.mean(nanpolicy='omit')
    FOV_mean     = (da - total_mean).mean(dim=('x','y'),nanpolicy='omit')
    noise        = FOV_mean.values - gaussian_filter(FOV_mean.values,gauss_window_std)
    params, _    = curve_fit(tides, time, FOV_mean.values - noise, p0=init_tides_guess)
    tidal_signal = tides(time, *params)
    FOV_pert     = da.values - total_mean.values - FOV_mean.values + noise - tidal_signal
    
    small_scales = np.zeros(da.shape)
    large_scales = np.zeros(da.shape)

    print('Start: Separation in large and small scales.')
    for t in tqdm(range(da.coords['time'].size)):
        small_scales[:,:,t], large_scales[:,:,t] = AMTM_filter(FOV_pert[:,:,t])

    decomp_ds = xr.Dataset(
        {
            "T tides": xr.DataArray(
                tidal_signal,
                dims=('time'),
                coords={'time': da['time'].values},
            ),
            "T large": xr.DataArray(
                large_scales,
                dims=('y', 'x', 'time'),
                coords={'time': da['time'].values, 'x': da['x'].values, 'y': da['y'].values},
            ),
            "T small": xr.DataArray(
                small_scales,
                dims=('y', 'x', 'time'),
                coords={'time': da['time'].values, 'x': da['x'].values, 'y': da['y'].values},
            ),
            "T noise": xr.DataArray(
                noise,
                dims=('time'),
                coords={'time': da['time'].values},
            ),
        },
        attrs={
            "instrument_name": "AMTM",
            "long_name": "Advanced Mesospheric Temperature Mapper",
            "location": 'Río Grande',
            "temporal_resolution [s]": 35,
            "horizontal_resolution [m]": 625,
            "filter": "subtraction of 2d linear fit and first harmonic",
            "contact": "robert.reichert@physik.uni-muenchen.de",
            "average Temperature [K]": T0,
            "amplitude 24h oscillation [K]": params[0],
            "phase 24h oscillation [rad]": params[1],
            "amplitude 12h oscillation [K]": params[2],
            "phase 12h oscillation [rad]": params[3],
            "amplitude 8h oscillation [K]": params[4],
            "phase 8h oscillation [rad]": params[5],
        },
    )

    print('AMTM Decomposition done!')
    
    return decomp_ds