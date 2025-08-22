import numpy as np
import itertools
from tqdm import tqdm

def kxky_2_lhtheta(kx,ky):
    """
    Convert the wave vector components into a wavelength and an orientation
    Keep in mind that arctan2() returns signed angles between [-np.pi,np.pi] defined from the positive x-axis
    while I defined my angles from [0,2*np.pi] going clockwise from the positive y-axis.
    """
    k     = np.sqrt(kx**2+ky**2)
    theta = np.arctan2(ky, kx)
    theta = np.pi/2-theta
    theta[theta<0] = 2*np.pi + theta[theta<0]
    
    return 2*np.pi/k, theta

def A_kx_ky(list_of_labels,CWT,segments):
    """
    Computes the wave paket properties such as wavelength, propagation direction and amplitude 
    as function of x and y based on the CWT and the labelled region within the WPS.

    Parameters
    ----------
    list_of_labels : list of ints
        label(s) of the wavepaket which properties should be computed.
    CWT : dict
        dictionary containing among other things the wavelet coefficients.
        dictionary is provided by juwavelet.transform.decompose()
    segments : array of ints
        an array of the same dimensions as the CWT['decomposition'] entry that marks clusters in the WPS
        and hence marks individual wave pakets.

    Returns
    -------
    3 x ndarrays
    """

    dim = CWT['decomposition'].shape

    A  = np.zeros(dim[2:4])
    kx = np.zeros(dim[2:4])
    ky = np.zeros(dim[2:4])

    T, P = np.meshgrid(CWT['theta'],CWT['period']*1e3)
    
    for i, j in tqdm(list(itertools.product(range(dim[2]), range(dim[3])))):
        mask = np.isin(segments[:,:,i,j], list_of_labels)
        if np.count_nonzero(mask) > 0:
            weights = np.abs(CWT["decomposition"][:,:,i,j]) ** 2
            if np.sum(weights[mask]) > 0:
                A[i,j] = np.sqrt(2)*np.sqrt(np.nanmax(weights[mask]))
                true_indices = np.argwhere(mask)
                max_index = np.argmax(weights[mask])  
                true_max_index = true_indices[max_index]  
                kx[i, j] = 2*np.pi/P[true_max_index[0], true_max_index[1]]*np.sin(T[true_max_index[0], true_max_index[1]])
                ky[i, j] = 2*np.pi/P[true_max_index[0], true_max_index[1]]*np.cos(T[true_max_index[0], true_max_index[1]])
                    
    return A, kx, ky

def A_kx_ky_kz(list_of_labels,XWT,segments,dz):
    """
    Computes the wave paket properties such as wavelength, propagation direction and amplitude 
    as function of x and y based on the CWT and the labelled region within the WPS.

    Parameters
    ----------
    list_of_labels : list of ints
        label(s) of the wavepaket which properties should be computed.
    CWT : dict
        dictionary containing among other things the wavelet coefficients.
        dictionary is provided by juwavelet.transform.decompose()
    segments : array of ints
        an array of the same dimensions as the CWT['decomposition'] entry that marks clusters in the WPS
        and hence marks individual wave pakets.

    Returns
    -------
    3 x ndarrays
    """

    dim = XWT['decomposition'].shape

    A  = np.zeros(dim[2:4])
    kx = np.zeros(dim[2:4])
    ky = np.zeros(dim[2:4])
    kz = np.zeros(dim[2:4])

    T, P = np.meshgrid(XWT['theta'],XWT['period']*1e3)
    
    for i, j in tqdm(list(itertools.product(range(dim[2]), range(dim[3])))):
        mask = np.isin(segments[:,:,i,j], list_of_labels)
        if np.count_nonzero(mask) > 0:
            weights = np.abs(XWT["decomposition"][:,:,i,j])
            phase = np.angle(XWT['decomposition'][:,:,i,j])
            if np.sum(weights[mask]) > 0:
                A[i,j] = np.sqrt(2)*np.sqrt(np.nanmax(weights[mask]))
                true_indices = np.argwhere(mask)
                max_index = np.argmax(weights[mask])  
                true_max_index = true_indices[max_index]  
                kx[i, j] = 2*np.pi/P[true_max_index[0], true_max_index[1]]*np.sin(T[true_max_index[0], true_max_index[1]])
                ky[i, j] = 2*np.pi/P[true_max_index[0], true_max_index[1]]*np.cos(T[true_max_index[0], true_max_index[1]])
                kz[i,j] = phase[true_max_index[0], true_max_index[1]]/dz
                
    # If kz<0 the wave vector direction is turned by 180°
    kx[kz<0]=-kx[kz<0]
    ky[kz<0]=-ky[kz<0]
    
    nan_mask=(A==0)
    A[nan_mask]=np.nan
    kx[nan_mask]=np.nan
    ky[nan_mask]=np.nan
    kz[nan_mask]=np.nan
                            
    return A, kx, ky, kz

def single_A_kx_ky(list_of_labels,CWT,x,y,segments):

    T, P, Y, X = np.meshgrid(CWT['theta'],CWT['period']*1e3,y,x)
    
    mask = np.isin(segments, list_of_labels)
    
    at_least_one_finite = np.any(mask, axis=(0, 1)) # is the axis rithg?
    size = np.sum(at_least_one_finite)
    
    if np.count_nonzero(mask) > 0:
        weights = np.abs(CWT['decomposition'])**2
        if np.sum(weights[mask]) > 0:
            A = np.sqrt(2)*np.sqrt(np.nanmax(weights[mask]))
            true_indices = np.argwhere(mask)
            max_index = np.argmax(weights[mask])  
            true_max_index = true_indices[max_index]  
            kx = 2*np.pi/P[true_max_index[0], true_max_index[1], true_max_index[2], true_max_index[3]]*np.sin(T[true_max_index[0], true_max_index[1], true_max_index[2], true_max_index[3]])
            ky = 2*np.pi/P[true_max_index[0], true_max_index[1], true_max_index[2], true_max_index[3]]*np.cos(T[true_max_index[0], true_max_index[1], true_max_index[2], true_max_index[3]])
            xpos = X[true_max_index[0], true_max_index[1], true_max_index[2], true_max_index[3]]
            ypos = Y[true_max_index[0], true_max_index[1], true_max_index[2], true_max_index[3]]
    
    return size, xpos, ypos, A, kx, ky

def single_A_kx_ky_kz(list_of_labels,XWT,x,y,segments,dz):

    T, P, Y, X = np.meshgrid(XWT['theta'],XWT['period']*1e3,y,x)
    
    mask = np.isin(segments, list_of_labels)
    
    at_least_one_finite = np.any(mask, axis=(0, 1)) # is the axis rithg?
    size = np.sum(at_least_one_finite)
    
    if np.count_nonzero(mask) > 0:
        weights = np.abs(XWT['decomposition'])
        phase = np.angle(XWT['decomposition'])
        if np.sum(weights[mask]) > 0:
            A = np.sqrt(2)*np.sqrt(np.nanmax(weights[mask]))
            true_indices = np.argwhere(mask)
            max_index = np.argmax(weights[mask])  
            true_max_index = true_indices[max_index]  
            kx = 2*np.pi/P[true_max_index[0], true_max_index[1], true_max_index[2], true_max_index[3]]*np.sin(T[true_max_index[0], true_max_index[1], true_max_index[2], true_max_index[3]])
            ky = 2*np.pi/P[true_max_index[0], true_max_index[1], true_max_index[2], true_max_index[3]]*np.cos(T[true_max_index[0], true_max_index[1], true_max_index[2], true_max_index[3]])
            kz = phase[true_max_index[0], true_max_index[1], true_max_index[2], true_max_index[3]]/dz
            xpos = X[true_max_index[0], true_max_index[1], true_max_index[2], true_max_index[3]]
            ypos = Y[true_max_index[0], true_max_index[1], true_max_index[2], true_max_index[3]]
    
    # If kz<0 the wave vector direction is turned by 180°
    if kz<0:
        kx=-kx
        ky=-ky
    
    return size, xpos, ypos, A, kx, ky, kz

def single_c_ci_m_wi_MF(A,kx,ky,f,wind,T0,N):
    
    # Define the gravity constant @ OH layer altitude
    g = 9.54 #[m/s^2]
    fwhm_OH = 7*1e3
    
    lh, theta = kxky_2_lhtheta(kx,ky)
        
    k_abs  = np.sqrt(kx**2+ky**2)
    c_abs  = np.abs(f)/k_abs
    cx, cy = c_abs*np.sin(theta), c_abs*np.cos(theta)
    
    doppler_x = (wind[0]*np.sin(theta) + wind[1]*np.cos(theta)) * np.sin(theta)
    doppler_y = (wind[0]*np.sin(theta) + wind[1]*np.cos(theta)) * np.cos(theta)
    
    cx_i, cy_i = cx-doppler_x, cy-doppler_y
    ci_abs = np.sqrt(cx_i**2+cy_i**2)
    
    # Derive the intrinsic frequency from the Doppler shift
    wi = ci_abs*k_abs #[1/s]
    
    m = compute_m(ci_abs,k_abs,T0,N)
    wi[np.isnan(m)]=np.nan
    
    # Define the correction factor due to phase averaging
    correction_factor = np.exp(-3.56*fwhm_OH**2/(2*np.pi/m)**2)
    
    # Define the momentum flux in propagation direction
    MF = g**2 * wi / 2 / N**3 * np.sqrt(1-(wi/N)**2) * (A/T0)**2 / correction_factor**2

    return [cx,cy], [cx_i,cy_i], m, wi, MF 

def compute_m(ci_abs,k_abs,T0,N,lz_cutoff=14*1e3):
    
    # Define the gravity constant @ OH layer altitude
    g = 9.54 #[m/s^2]
    R = 287 #[J/kg/K]
    H = R*T0/g #[m]
    
    # Derive the vertical wavenumber from the dispersion relation
    m = np.sqrt(N**2/ci_abs**2-1/4/H**2-k_abs**2)
    m[m==0]=np.nan
    m[m>(2*np.pi/lz_cutoff)]=np.nan

    return m