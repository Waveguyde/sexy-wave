import numpy as np

def m_squared(N,ci,H,k,u_prime,u_prime2):
    """
    This function returns the squared vertical wavenumber (Taylor Goldstein equation);
    Make sure that u_prime and u_prime2 are the wind shear and wind curvature in the direction of wave propagation;
    """
    return N**2/ci**2-1/4/H**2-k**2-1/H*u_prime/ci+u_prime2/ci

def intrinsic_phasespeed(u,v,cx,cy):
    """
    Given the observed phase speed and background horizontal wind speed, this function returns the absolute value of the intrinsic phase speed;
    """
    cx_i = cx - u
    cy_i = cy - v
    return np.sqrt(cx_i**2+cy_i**2)

def wind_shear(du_dz,dv_dz,cx,cy):
    """
    This function computes the projection of wind shear onto the wave propagation direction;
    """
    c = np.sqrt(cx**2+cy**2)
    return (du_dz*cx + dv_dz*cy)/c

def wind_curvature(du2_dz2,dv2_dz2,cx,cy):
    """
    This function computes the projection of wind curvature onto the wave propagation direction;
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
    correction_factor[np.isnan(correction_factor)]=1.
    T_true = A/correction_factor
    return g**2 * omega_i / 2 / N**3 * np.sqrt(1-(omega_i/N)**2) * (T_true/T0)**2