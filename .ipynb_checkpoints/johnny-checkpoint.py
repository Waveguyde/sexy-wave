import sys
sys.path.append('/home/r/Robert.Reichert/juwavelet')
import juwavelet.transform as transform

import numpy as np
import pandas as pd
import xarray as xr
import copy
import datetime

from scipy.ndimage import gaussian_filter
from skimage.segmentation import watershed
from skimage.morphology import local_minima
from skimage.measure import label

def synchronize(*datasets):
    """Synchronize multiple datasets to the overlapping time range."""
    
    if not datasets:
        raise ValueError("At least one dataset must be provided.")

    # Step 1: Find the global start and end time
    start_time = max(ds['time'].min().values for ds in datasets)
    end_time = min(ds['time'].max().values for ds in datasets)

    # Step 2: Truncate all datasets to the overlapping time range
    synchronized_datasets = [ds.sel(time=slice(start_time, end_time)) for ds in datasets]

    return synchronized_datasets

def get_CORAL_time(reference_datetime_object,seconds_since_reference):
    time=[]
    for s in seconds_since_reference:
        try:
            s=int(s)
            dt=reference_datetime_object+datetime.timedelta(seconds=s)
            time.append(pd.Timestamp(dt))
        except (ValueError, TypeError):
            print('Skipping invalid value: {ms}')
    return time

def get_SAAMER_time(reference_datetime_object,days_since_reference):
    time=[]
    for d in days_since_reference:
        try:
            dt=reference_datetime_object+datetime.timedelta(days=d)
            time.append(pd.Timestamp(dt))
        except (ValueError, TypeError):
            print('Skipping invalid value: {ms}')
    return time

def compute_N(ds):
    
    Me = 5.9722*1e24
    G  = 6.6743*1e-11
    Re = 6.371*1e6
    cp = 1003.5 #[J/kg/K]
    
    dim = ds['temperature'].shape
    
    dz = ds.coords['altitude'].diff(dim='altitude').values[0]

    # Unit detection: Assume km if dz < 10, else meters
    if np.abs(dz) < 10:  
        dz *= 1e3           # Convert dz to meters
        ds['altitude'] = ds['altitude']*1e3     # Convert altitude to meters
    
    if len(dim)==1:
        T = ds['temperature']
        sigma_T = ds['temperature_err']
        dT_dz = np.gradient(T, dz)
        sigma_dT_dz = np.sqrt(2) * sigma_T / dz

        g         = G*Me/(Re+ds['altitude'].values)**2
        N         = np.sqrt(g/T*(dT_dz+g/cp))
        sigma_N   = (1 / (2 * N)) * np.sqrt((g * (dT_dz + g / cp) / T**2)**2 * sigma_T**2 + (g / T)**2 * sigma_dT_dz**2) 

        N_ds = xr.Dataset(
            {
                "N": xr.DataArray(
                    N,
                    dims=("altitude"),
                    coords={"altitude": ds['altitude'].values},
                ),
                "N_err": xr.DataArray(
                    sigma_N,
                    dims=("altitude"),
                    coords={"altitude": ds['altitude'].values},
                ),
            },
        )

    if len(dim)==2:
        T = ds['temperature']
        sigma_T = ds['temperature_err']
        dT_dz = np.gradient(T, dz, axis=1)
        sigma_dT_dz = np.sqrt(2) * sigma_T / dz
    
        g         = G*Me/(Re+ds['altitude'].values)**2
        N         = np.sqrt(g/T*(dT_dz+g/cp))
        sigma_N   = (1 / (2 * N)) * np.sqrt((g * (dT_dz + g / cp) / T**2)**2 * sigma_T**2 + (g / T)**2 * sigma_dT_dz**2) 

        N_ds = xr.Dataset(
            {
                "N": xr.DataArray(
                    N,
                    dims=("time", "altitude"),
                    coords={"time": ds['time'].values, "altitude": ds['altitude'].values},
                ),
                "N_err": xr.DataArray(
                    sigma_N,
                    dims=("time", "altitude"),
                    coords={"time": ds['time'].values, "altitude": ds['altitude'].values},
                ),
            },
        )
    
    return N_ds

def OH_gaussian_kernel(z, center=86.8, sigma=8.6/2.355):
    """Create a 1D Gaussian kernel with the given size and standard deviation (sigma)."""
    dz = np.diff(z)[0]

    # Unit detection: Assume m if dz > 10, else kilometers
    if np.abs(dz) > 10:  
        z = z*1e-3     # Convert altitude to kilometers
        
    kernel = np.exp(-0.5 * ((z - center) / sigma) ** 2)
    return kernel / np.sum(kernel)  # Normalize so the sum is 1

def OH_layer_kernel(z, a, b, c, d, e):
    """Create a 1D OH layer kernel with the given parameters."""
    kernel = a * np.exp(-b*(z-c)+d*np.exp(-e*(z-c)))
    return kernel / np.sum(kernel)  # Normalize so the sum is 1

def weighted_circmean(theta, weights):
    """
    Computes a weighted mean of angles and takes into account the periodicity of angles.

    Parameters
    ----------
    theta : ndarray
        angles to be averaged
    weights : ndarray
        weights of the angles

    Returns
    -------
    ndarray
    
    Comment: np.arctan2() returns a signed angle with respect to the positive x-axis. 
    Example: 
    x = np.array([-1, +1, +1, -1])
    y = np.array([-1, -1, +1, +1])
    np.arctan2(y, x) * 180 / np.pi
    array([-135.,  -45.,   45.,  135.])
    """
    x0, y0 = np.sin(theta), np.cos(theta)
    x, y   = np.average(x0, weights = weights), np.average(y0, weights = weights)
    theta_bar = np.arctan2(y, x)
    theta_bar = np.pi/2 - theta_bar #This step is to convert the range of angles [-np.pi/2,np.pi/2] to [0,np.pi].
    
    return theta_bar

def wavefield_segmentation(data, sigma):
    """
    Labels individual segments of a given dataset.

    Parameters
    ----------
    data : ndarray
        data to be segmented
    sigma : list of floats
        tuple of sigma values for the gaussian filter

    Returns
    -------
    ndarray
    """
    
    idata = np.max(data) - data # Invert the data
    connection_param = len(data.shape)
    
    # First smooth the data in order to get rid of tiny minima and reduce the number of segments
    smoo_idata=gaussian_filter(idata,sigma=sigma)
    
    # Define the minima to be used by watershed segmentation
    mini = local_minima(smoo_idata,connectivity=connection_param,allow_borders=False)
    minima_pos = label(mini,connectivity=connection_param)
    
    return watershed(idata,connectivity=connection_param,markers=minima_pos)

def wavepaket_reconstruction(list_of_labels,CWT,segments):
    """
    Reconstructs a wavepaket based on the CWT and the labelled region within the WPS.
    wavefield_segmentation() must be done prior to the reconstruction.

    Parameters
    ----------
    list_of_labels : list of ints
        label(s) of the wavepaket that should be reconstructed
    CWT : dict
        dictionary containing among other things the wavelet coefficients to perform the reconstruction.
        dictionary is provided by juwavelet.transform.decompose()
    segments : array of ints
        an array of the same dimensions as the CWT['decomposition'] entry that marks clusters in the WPS
        and hence marks individual wave pakets.

    Returns
    -------
    ndarray
    """
    CWT_copy = copy.deepcopy(CWT)
    mask = np.isin(segments, list_of_labels)
    CWT_copy["decomposition"][~mask] = 0
    
    return transform.reconstruct2d(CWT_copy)

def CWT2XWT(cwt1,cwt2):
    
    dim=cwt1['decomposition'].shape
    
    if len(dim) == 4:
        XWT={
            "decomposition": cwt1['decomposition']*np.conjugate(cwt2['decomposition']),
            "dx": cwt1['dx'],
            "dy": cwt1['dy'],
            "dj": cwt1['dj'],
            "js": cwt1['js'],
            "jt": cwt1['jt'],
            "scale": cwt1['scale'],
            "theta": cwt1['theta'],
            "period": cwt1['period'],
            "aspect": cwt1['aspect'],
            "mode": cwt1['mode'],
            "opts": cwt1['opts']}
    
    if len(dim) == 2:
        XWT={
            "decomposition": cwt1['decomposition']*np.conjugate(cwt2['decomposition']),
            "dx": cwt1['dx'],
            "dj": cwt1['dj'],
            "js": cwt1['js'],
            "scale": cwt1['scale'],
            "period": cwt1['period'],
            "mode": cwt1['mode'],
            "opts": cwt1['opts']}
    
    return XWT
    
#def truncate_to_valid_range(ds, var='temperature', dim='time'):
#    """Find the start and end indices of valid (non-NaN) data for a given dimension."""
#    var_data = ds[var]
#    valid_data = (~np.all(np.isnan(var_data), axis=1)).values
#    start = np.argmax(valid_data)
#    end = len(valid_data) - np.argmax(valid_data[::-1]) - 1
#    return ds.isel({dim: slice(start, end + 1)})
    
def truncate_to_valid_range(ds_or_da, var='temperature', dim='time'):
    """Find the start and end indices of valid (non-NaN) data for a given dimension.
    
    Supports both xarray.Dataset and xarray.DataArray as input.
    
    Parameters:
        ds_or_da (xarray.Dataset or xarray.DataArray): Input dataset or data array.
        var (str, optional): Variable name (used only if input is a Dataset). Defaults to 'temperature'.
        dim (str, optional): Dimension along which to truncate. Defaults to 'time'.
    
    Returns:
        xarray.Dataset or xarray.DataArray: Truncated version of input with only valid data.
    """
    
    if isinstance(ds_or_da, xr.Dataset):
        var_data = ds_or_da[var]
    elif isinstance(ds_or_da, xr.DataArray):
        var_data = ds_or_da
    else:
        raise TypeError("Input must be an xarray Dataset or DataArray")

    # Determine validity across the selected dimension
    if var_data.ndim > 1:
        valid_data = (~np.all(np.isnan(var_data), axis=tuple(i for i in range(var_data.ndim) if i != var_data.get_axis_num(dim)))).values
    else:
        valid_data = ~np.isnan(var_data).values

    # Find first and last valid index
    start = np.argmax(valid_data)
    end = len(valid_data) - np.argmax(valid_data[::-1]) - 1

    return ds_or_da.isel({dim: slice(start, end + 1)})

def CORAL_preprocessing(path,reference_date):

    coral_ds                    = xr.open_dataset(path,decode_times=False)
    coral_ds['temperature']     = xr.where(coral_ds['temperature']!=0,coral_ds['temperature'],np.nan)
    coral_ds['temperature_err'] = xr.where(coral_ds['temperature_err']!=0,coral_ds['temperature_err'],np.nan)

    # Apply truncation to the coral_temp dataset
    coral_ds = truncate_to_valid_range(coral_ds)

    seconds_since_reference = coral_ds.time*1e-3
    coral_ds['time']        = get_CORAL_time(reference_date,seconds_since_reference)

    coral_temp_ds = xr.Dataset(
        {
            "temperature": xr.DataArray(
                coral_ds['temperature'].values,
                dims=("time", "altitude"),
                coords={"time": coral_ds['time'].values, "altitude": coral_ds['altitude'].values},
            ),
            "temperature_err": xr.DataArray(
                coral_ds['temperature_err'].values,
                dims=("time", "altitude"),
                coords={"time": coral_ds['time'].values, "altitude": coral_ds['altitude'].values},
            ),
        },
    )

    return coral_temp_ds

def SAAMER_preprocessing(path,reference_date,start,end):

    saamer_ds            = xr.open_dataset(path,engine='netcdf4')
    days_since_reference = saamer_ds.time.values
    saamer_ds['time']    = get_SAAMER_time(reference_date,days_since_reference)
    saamer_ds            = saamer_ds.sel(time=slice(start,end))

    u = np.flip(saamer_ds['zonal_wind'].values,axis=0)
    v = np.flip(saamer_ds['meridional_wind'].values,axis=0)
    du_dz = np.gradient(u,axis=0)*1e-3
    dv_dz = np.gradient(v,axis=0)*1e-3
    du2_dz2 = np.gradient(du_dz,axis=0)*1e-3
    dv2_dz2 = np.gradient(dv_dz,axis=0)*1e-3

    # Create xarray Dataset with explicit variable names
    saamer_wind_ds = xr.Dataset(
        {
            "u": (["altitude", "time"], u),
            "v": (["altitude", "time"], v),
            "du_dz": (["altitude", "time"], du_dz),
            "dv_dz": (["altitude", "time"], dv_dz),
            "du2_dz2": (["altitude", "time"], du2_dz2),
            "dv2_dz2": (["altitude", "time"], dv2_dz2),
            "u_err": (["altitude", "time"], np.full_like(u,3.0)),
            "v_err": (["altitude", "time"], np.full_like(u,3.0)),
            "du_dz_err": (["altitude", "time"], np.full_like(u,np.sqrt(2)*3.0*1e-3)),
            "dv_dz_err": (["altitude", "time"], np.full_like(u,np.sqrt(2)*3.0*1e-3)),
            "du2_dz2_err": (["altitude", "time"], np.full_like(u,2*3.0*1e-6)),
            "dv2_dz2_err": (["altitude", "time"], np.full_like(u,2*3.0*1e-6)),
        },
        coords={"time": saamer_ds['time'].values, "altitude": saamer_ds['altitude'].values}
    )
    
    return saamer_wind_ds

def high2low_by_binning(ds1,ds2):
    """
    Synchronize two datasets to the overlapping time range and match the lower resolution 
    by binning and averaging the higher-resolution dataset.
    """

    # Step 1: Find the overlapping time range (Moved here)
    start_time = max(ds1['time'].min().values, ds2['time'].min().values)
    end_time = min(ds1['time'].max().values, ds2['time'].max().values)
    
    ds1 = ds1.sel(time=slice(start_time, end_time))
    ds2 = ds2.sel(time=slice(start_time, end_time))
    
    # Step 2: Identify the dataset with the lower resolution
    time_diff_1 = np.diff(ds1['time']).astype('timedelta64[s]').astype(float)
    time_step_1 = np.median(time_diff_1)  # Now this is a float (in seconds)
    if time_step_1 != np.mean(time_diff_1):
        print('Irregular time sampling in ds1!')

    time_diff_2 = np.diff(ds2['time']).astype('timedelta64[s]').astype(float)
    time_step_2 = np.median(time_diff_2)  # Now this is a float (in seconds)
    if time_step_2 != np.mean(time_diff_2):
        print('Irregular time sampling in ds2!')
    
    if time_step_1 <= time_step_2:
        low_res_ds, high_res_ds = ds2, ds1
        time_step_low, time_step_high = time_step_2, time_step_1
        # Convert to Pandas-friendly frequency string
        time_step_str = f"{int(time_step_2)}S"
    else:
        low_res_ds, high_res_ds = ds1, ds2
        # Convert to Pandas-friendly frequency string
        time_step_low, time_step_high = time_step_1, time_step_2
        time_step_str = f"{int(time_step_1)}S"
    
    # Step 3: Resample high-resolution dataset using computed time step
    high_res_binned = high_res_ds.resample(time=time_step_str).mean(skipna=True)  # Prevent NaN
    for var in high_res_ds.data_vars:
        if var.endswith("_err"):
            mean_var = var.removesuffix("_err")
            high_res_binned[var] = np.sqrt(high_res_binned[var]**2/(time_step_high/time_step_low)+high_res_ds[mean_var].resample(time=time_step_str).var(skipna=True))    

    # Step 4: Adjust for half-bin shift
    high_res_binned['time'] = high_res_binned['time'] + np.timedelta64(int(time_step_low // 2), 's')
    #high_res_binned = high_res_binned.interp(time=low_res_ds["time"], kwargs={"fill_value": 'extrapolate'}, method='cubic')
    mean_values = {var: high_res_binned[var].mean(dim="time", skipna=True) for var in high_res_binned.data_vars}
    interpolated_vars = {
        var: high_res_binned[var].interp(time=low_res_ds["time"], kwargs={"fill_value": mean_values[var].values}, method='nearest')
        for var in high_res_binned.data_vars
    }
    high_res_binned = xr.Dataset(interpolated_vars)

    return low_res_ds, high_res_binned

def find_connected_low_values(array, start_coord, threshold=1):
    """
    Find all connected coordinates where values are below a threshold,
    starting from an initial coordinate.

    Parameters:
        array (np.ndarray): 2D array of values.
        start_coord (tuple): (row, col) coordinate where the search starts.
        threshold (float): Value threshold to determine valid regions.

    Returns:
        List of (row, col) coordinates that are connected and meet the threshold condition.
    """
    from collections import deque

    rows, cols = array.shape
    r0, c0 = start_coord
    
    if array[r0, c0] >= threshold:
        return []  # If the start coordinate doesn't meet the condition, return empty

    visited = set()
    queue = deque([start_coord])
    result = []

    directions = [(-1, 0), (1, 0), (0, -1), (0, 1), # Up, Down, Left, Right
                 (-1,-1), (1,1), (-1,1), (1,-1)] # diagonal 

    while queue:
        r, c = queue.popleft()
        
        if (r, c) in visited:
            continue
        visited.add((r, c))
        result.append((r, c))

        for dr, dc in directions:
            nr, nc = r + dr, c + dc
            if 0 <= nr < rows and 0 <= nc < cols:  # Check boundaries
                if (nr, nc) not in visited and array[nr, nc] < threshold:
                    queue.append((nr, nc))

    return result

def find_local_minimum(array, start_coord):
    """
    Find a local minimum in a 2D array starting from a given coordinate.
    
    Parameters:
        array (np.ndarray): 2D input array.
        start_coord (tuple): (row, col) starting coordinate.
    
    Returns:
        tuple: Coordinates (row, col) of the local minimum.
    """
    rows, cols = array.shape
    r, c = start_coord

    while True:
        # Get neighbors (up, down, left, right, and diagonals)
        neighbors = []
        for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1), (-1, -1), (-1, 1), (1, -1), (1, 1)]:
            nr, nc = r + dr, c + dc
            if 0 <= nr < rows and 0 <= nc < cols:  # Check boundaries
                neighbors.append((nr, nc, array[nr, nc]))
        
        # Find the neighbor with the smallest value
        min_neighbor = min(neighbors, key=lambda x: x[2])  # (row, col, value)

        # If the current position is smaller than all neighbors, stop
        if array[r, c] <= min_neighbor[2]:
            return (r, c)

        # Move to the new minimum neighbor
        r, c = min_neighbor[:2]