import numpy as np
import itertools
from tqdm import tqdm

def kxky_2_lhtheta(kx,ky):
    k=np.sqrt(kx**2+ky**2)
    theta = np.arctan2(ky, kx)
    theta = np.pi/2-theta
    theta[theta<0]=2*np.pi+theta[theta<0]
    return 2*np.pi/k, theta    

def A_kx_ky(CWT):
    """
    Finds the maximum amplitude in the WPS and corresponding kx and ky for each pair (x,y). 

    Parameters
    ----------
    CWT : dict
        dictionary containing among other things the wavelet coefficients.
        dictionary is provided by juwavelet.transform.decompose2d()

    Returns
    -------
    3 x ndarrays
    """
    
    dim = CWT['decomposition'].shape

    A  = np.zeros(dim[2:4])
    kx = np.zeros(dim[2:4])
    ky = np.zeros(dim[2:4])

    T, P = np.meshgrid(CWT['theta'],CWT['period'])
    
    for i, j in tqdm(list(itertools.product(range(dim[2]), range(dim[3])))):
        WPS = np.abs(CWT["decomposition"][:,:,i,j]) ** 2
        A[i,j] = np.sqrt(np.max(WPS))                
        max_index = np.argmax(WPS)  
        max_index_2d = np.unravel_index(max_index, WPS.shape)
        kx[i, j] = 2*np.pi/P[max_index_2d]*np.sin(T[max_index_2d])
        ky[i, j] = 2*np.pi/P[max_index_2d]*np.cos(T[max_index_2d])
                            
    return A, kx, ky

def A_kx_ky_kz(XWT,dz):
    """
    Finds the maximum amplitude in the XWS and corresponding kx, ky, and kz for each pair (x,y). 

    Parameters
    ----------
    XWT : dict
        dictionary containing among other things the wavelet coefficients and cross-wavelet coefficients.
        dictionary is provided by juwavelet.transform.decompose2d()
    dz : float
        increment in z

    Returns
    -------
    3 x ndarrays
    """
    
    dim = XWT['decomposition'].shape

    A  = np.zeros(dim[2:4])
    kx = np.zeros(dim[2:4])
    ky = np.zeros(dim[2:4])
    kz = np.zeros(dim[2:4])

    T, P = np.meshgrid(XWT['theta'],XWT['period'])
    
    for i, j in tqdm(list(itertools.product(range(dim[2]), range(dim[3])))):
        WPS = np.abs(XWT["decomposition"][:,:,i,j])
        phase = np.angle(XWT["decomposition"][:,:,i,j])
        A[i,j] = np.sqrt(np.max(WPS))                
        max_index = np.argmax(WPS)  
        max_index_2d = np.unravel_index(max_index, WPS.shape)
        kx[i, j] = 2*np.pi/P[max_index_2d]*np.sin(T[max_index_2d])
        ky[i, j] = 2*np.pi/P[max_index_2d]*np.cos(T[max_index_2d])
        kz[i, j] = phase[max_index_2d]/dz
        
    # If kz<0 the wave vector direction is turned by 180° 
    kx[kz<0]=-kx[kz<0]# ONLY VALID IF THE THIRD DIMENSION IS TIME!!!
    ky[kz<0]=-ky[kz<0]
    
    nan_mask=(A==0)
    A[nan_mask]=np.nan
    kx[nan_mask]=np.nan
    ky[nan_mask]=np.nan
    kz[nan_mask]=np.nan
                            
    return A, kx, ky, kz