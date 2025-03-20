import numpy as np

def m_squared(N,ci,H,k,u_prime,u_prime2):
    """
    This function returns the squared vertical wavenumber;
    Make sure that u_prime and u_prime2 are the wind shear and curvature in the direction of wave propagation;
    """
    return N**2/ci**2-1/4/H**2-k**2-1/H*u_prime/ci+u_prime2/ci

def intrinsic_phasespeed(u,v,cx,cy):
    """
    This function returns the Doppler shifted intrinsic phase speed;
    """
    #c_abs = np.sqrt(cx**2+cy**2)
    #doppler_wind = (u*cx/c_abs + v*cy/c_abs)
    cx_i = cx - u
    cy_i = cy - v
    return np.sqrt(cx_i**2+cy_i**2)#c_abs-doppler_wind

def wind_shear(du_dz,dv_dz,cx,cy):
    """
    This function returns the wind shear in the direction of wave propagation;
    """
    c = np.sqrt(cx**2+cy**2)
    return (du_dz*cx + dv_dz*cy)/c

def wind_curvature(du2_dz2,dv2_dz2,cx,cy):
    """
    This function returns the wind curvature in the direction of wave propagation;
    """
    c = np.sqrt(cx**2+cy**2)
    return (du2_dz2*cx + dv2_dz2*cy)/c

def momentum_flux(N,T0,A,fwhm,m2,omega_i):
    """
    This function returns the GW momentum flux;
    """
    g=9.54 
    lz = 2*np.pi/np.sqrt(m2)
    correction_factor = np.exp(-3.56*fwhm**2/lz**2)
    T_true = A/correction_factor
    return g**2 * omega_i / 2 / N**3 * np.sqrt(1-(omega_i/N)**2) * (T_true/T0)**2