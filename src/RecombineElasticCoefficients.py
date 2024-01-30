import torch
import warnings

def realsphvectors2voigt(asym: torch.Tensor, sym: torch.Tensor) -> torch.Tensor:
    """
    Recombines the spherical harmonic coefficients into a Voigt tensor

    Args:
        asym: The asymmetric coefficients
        sym: The symmetric coefficients
    """
    if asym.shape[-1] != 6 or sym.shape[-1] != 15:
        raise Exception("Unknown shape of spherical harmonic coefficients.")
    realsph = recombine_realsph_vecs(asym, sym)
    sph = realsph2sph(realsph)
    return sph2voigt(sph)

def sph2voigt(sph):
    msph = sph2msph(sph)
    jz = msph2jz(msph)
    cart = jz2cart(jz)
    voigt = cart2voigt(cart)
    return voigt

def recombine_realsph_vecs(asym, sym):
    if asym.shape[-1] != 6 or sym.shape[-1] != 15:
        raise Exception("Unknown shape of spherical harmonic coefficients.")
    realsph = {
        'c_0_0': asym[:, 0], 
        'c_2_-2': asym[:, 1],
        'c_2_-1': asym[:, 2],
        'c_2_0': asym[:, 3],
        'c_2_1': asym[:, 4],
        'c_2_2': asym[:, 5],
        's_0_0': sym[:, 0], 
        's_2_-2': sym[:, 1],
        's_2_-1': sym[:, 2],
        's_2_0': sym[:, 3],
        's_2_1': sym[:, 4],
        's_2_2': sym[:, 5],
        's_4_-4': sym[:, 6], 
        's_4_-3': sym[:, 7],
        's_4_-2': sym[:, 8],
        's_4_-1': sym[:, 9],
        's_4_0': sym[:, 10],
        's_4_1': sym[:, 11],
        's_4_2': sym[:, 12],
        's_4_3': sym[:, 13],
        's_4_4': sym[:, 14]
    }

    return realsph

### Convert real harmonic coefficients to complex spherical harmonic coefficients
def realsph2sph(realsph):
    sph = {}
    sph['s_0^0'] = realsph['s_0_0'] 
    sph['c_0^0'] = realsph['c_0_0'] 
    
    
    sph['s_2^-2'] = (2**(-1/2))*(realsph['s_2_2']-1j*realsph['s_2_-2'])
    sph['c_2^-2'] = (2**(-1/2))*(realsph['c_2_2']-1j*realsph['c_2_-2'])
    
    sph['s_2^-1'] = (2**(-1/2))*(realsph['s_2_1']-1j*realsph['s_2_-1'])
    sph['c_2^-1'] = (2**(-1/2))*(realsph['c_2_1']-1j*realsph['c_2_-1'])
    
    sph['s_2^0'] = realsph['s_2_0'] 
    sph['c_2^0'] = realsph['c_2_0'] 
    
    sph['s_2^1'] = -(1/(2**(1/2)))*(realsph['s_2_1']+1j*realsph['s_2_-1'])
    sph['c_2^1'] = -(1/(2**(1/2)))*(realsph['c_2_1']+1j*realsph['c_2_-1'])
    
    sph['s_2^2'] = (1/(2**(1/2)))*(realsph['s_2_2']+1j*realsph['s_2_-2'])
    sph['c_2^2'] = (1/(2**(1/2)))*(realsph['c_2_2']+1j*realsph['c_2_-2'])

    
    sph['s_4^-4'] = (1/(2**(1/2)))*(realsph['s_4_4']-1j*realsph['s_4_-4'])
    sph['s_4^-3'] = (1/(2**(1/2)))*(realsph['s_4_3']-1j*realsph['s_4_-3'])
    sph['s_4^-2'] = (1/(2**(1/2)))*(realsph['s_4_2']-1j*realsph['s_4_-2'])
    sph['s_4^-1'] = (1/(2**(1/2)))*(realsph['s_4_1']-1j*realsph['s_4_-1'])
    sph['s_4^0'] = realsph['s_4_0'] 
    sph['s_4^1'] = (-1/(2**(1/2)))*(realsph['s_4_1']+1j*realsph['s_4_-1'])
    sph['s_4^2'] = (1/(2**(1/2)))*(realsph['s_4_2']+1j*realsph['s_4_-2'])
    sph['s_4^3'] = (-1/(2**(1/2)))*(realsph['s_4_3']+1j*realsph['s_4_-3'])
    sph['s_4^4'] = (1/(2**(1/2)))*(realsph['s_4_4']+1j*realsph['s_4_-4'])

    return sph

## Convert conventional norm to mochizuki 
def sph2msph(sph):
    msph = {
        '*c_0^0' : ((1/(4*torch.pi))**(-1/2))*sph['c_0^0'],
        '*s_0^0' : ((1/(4*torch.pi))**(-1/2))*sph['s_0^0'],
        '*c_2^-2' : ((5/(4*torch.pi))**(-1/2))*sph['c_2^-2'],
        '*c_2^-1' : ((5/(4*torch.pi))**(-1/2))*sph['c_2^-1'],
        '*c_2^0' : ((5/(4*torch.pi))**(-1/2))*sph['c_2^0'],
        '*c_2^1' : ((5/(4*torch.pi))**(-1/2))*sph['c_2^1'],
        '*c_2^2' : ((5/(4*torch.pi))**(-1/2))*sph['c_2^2'],
        '*s_2^-2' : ((5/(4*torch.pi))**(-1/2))*sph['s_2^-2'],
        '*s_2^-1' : ((5/(4*torch.pi))**(-1/2))*sph['s_2^-1'],
        '*s_2^0' : ((5/(4*torch.pi))**(-1/2))*sph['s_2^0'],
        '*s_2^1' : ((5/(4*torch.pi))**(-1/2))*sph['s_2^1'],
        '*s_2^2' : ((5/(4*torch.pi))**(-1/2))*sph['s_2^2'],
        '*s_4^-4' : ((9/(4*torch.pi))**(-1/2))*sph['s_4^-4'],
        '*s_4^-3' : ((9/(4*torch.pi))**(-1/2))*sph['s_4^-3'],
        '*s_4^-2' : ((9/(4*torch.pi))**(-1/2))*sph['s_4^-2'],
        '*s_4^-1' : ((9/(4*torch.pi))**(-1/2))*sph['s_4^-1'],
        '*s_4^0' : ((9/(4*torch.pi))**(-1/2))*sph['s_4^0'],
        '*s_4^1' : ((9/(4*torch.pi))**(-1/2))*sph['s_4^1'],
        '*s_4^2' : ((9/(4*torch.pi))**(-1/2))*sph['s_4^2'],
        '*s_4^3' : ((9/(4*torch.pi))**(-1/2))*sph['s_4^3'],
        '*s_4^4' : ((9/(4*torch.pi))**(-1/2))*sph['s_4^4'],    
    }
    return msph

def msph2jz(msph):
    jz = torch.zeros([msph["*c_0^0"].shape[0],3,3,3,3], dtype=torch.complex64, device=msph["*c_0^0"].device)
    p=0
    z=1
    m=2
    jz[:, z, z, z, z] = (1/9)*msph['*c_0^0']+(4/45)*msph['*s_0^0']+(4/9)*msph['*c_2^0']+(8/63)*msph['*s_2^0']+(8/35)*msph['*s_4^0']
    
    jz[:, p, p, z, z] = (1/3)*(2/3)**(1/2)*msph['*c_2^2']+(-4/21)*(2/3)**(1/2)*msph['*s_2^2']+(2/7)*(2/5)**(1/2)*msph['*s_4^2']
    jz[:, z, z, p, p] = jz[:, p, p, z, z]
    
    jz[:, p, p, m, m] = (2/15)*msph['*s_0^0']-(4/21)*msph['*s_2^0']+(2/35)*msph['*s_4^0']
    jz[:, m, m, p, p] = jz[:, p, p, m, m]
    
    jz[:, p, p, z, m] = (-2/7)*(1/3)**(1/2)*msph['*s_2^1']+(1/7)*(2/5)**(1/2)*msph['*s_4^1']
    jz[:, p, p, m, z] = jz[:, p, p, z, m]
    jz[:, m, z, p, p] = jz[:, p, p, z, m]
    jz[:, z, m, p, p] = jz[:, p, p, z, m]

    jz[:, p, p, p, m] = (-1/3)*(2/3)**(1/2)*msph['*c_2^2']+(-2/21)*(2/3)**(1/2)*msph['*s_2^2']+(1/7)*(2/5)**(1/2)*msph['*s_4^2']
    jz[:, p, p, m, p] = jz[:, p, p, p, m]
    jz[:, p, m, p, p] = jz[:, p, p, p, m]
    jz[:, m, p, p, p] = jz[:, p, p, p, m]
    
    jz[:, p, p, p, z] = (2/35)**(1/2)*msph['*s_4^3']
    jz[:, p, p, z, p] = jz[:, p, p, p, z]
    jz[:, p, z, p, p] = jz[:, p, p, p, z]
    jz[:, z, p, p, p] = jz[:, p, p, p, z]
    
    jz[:, p, p, p, p] = 2*(2/35)**(1/2)*msph['*s_4^4']

    jz[:, z, z, m, m] = (1/3)*(2/3)**(1/2)*msph['*c_2^-2']+(-4/21)*(2/3)**(1/2)*msph['*s_2^-2']+(2/7)*(2/5)**(1/2)*msph['*s_4^-2']
    jz[:, m, m, z, z] = jz[:, z, z, m, m] 

    jz[:, z, z, z, m] = (1/3)*(1/3)**(1/2)*msph['*c_2^-1']+(2/21)*(1/3)**(1/2)*msph['*s_2^-1']+(2/7)*(2/5)**(1/2)*msph['*s_4^-1']
    jz[:, z, z, m, z] = jz[:, z, z, z, m]
    jz[:, z, m, z, z] = jz[:, z, z, z, m]
    jz[:, m, z, z, z] = jz[:, z, z, z, m]
    
    jz[:, z, z, p, m] = (-1/9)*msph['*c_0^0']+(-1/9)*msph['*c_2^0'] +(2/45)*msph['*s_0^0']+(4/63)*msph['*s_2^0']+(4/35)*msph['*s_4^0']
    jz[:, z, z, m, p] = jz[:, z, z, p, m]
    jz[:, p, m, z, z] = jz[:, z, z, p, m]
    jz[:, m, p, z, z] = jz[:, z, z, p, m]
    
    jz[:, z, z, p, z] = (1/3)*(1/3)**(1/2)*msph['*c_2^1']+(2/21)*(1/3)**(1/2)*msph['*s_2^1']+(2/7)*(2/5)**(1/2)*msph['*s_4^1']
    jz[:, z, z, z, p] = jz[:, z, z, p, z]
    jz[:, p, z, z, z] = jz[:, z, z, p, z]
    jz[:, z, p, z, z] = jz[:, z, z, p, z]
    
    jz[:, m, m, m, m] = 2*(2/35)**(1/2)*msph['*s_4^-4']
    
    jz[:, m, m, z, m] = (2/35)**(1/2)*msph['*s_4^-3']
    jz[:, m, m, m, z] = jz[:, m, m, z, m]
    jz[:, z, m, m, m] = jz[:, m, m, z, m]
    jz[:, m, z, m, m] = jz[:, m, m, z, m]
    
    jz[:, m, m, p, m] = (-1/3)*(2/3)**(1/2)*msph['*c_2^-2']+(-2/21)*(2/3)**(1/2)*msph['*s_2^-2']+(1/7)*(2/5)**(1/2)*msph['*s_4^-2']
    jz[:, m, m, m, p] = jz[:, m, m, p, m]
    jz[:, p, m, m, m] = jz[:, m, m, p, m]
    jz[:, m, p, m, m] = jz[:, m, m, p, m]

    jz[:, m, m, p, z] = (-2/7)*(1/3)**(1/2)*msph['*s_2^-1']+(1/7)*(2/5)**(1/2)*msph['*s_4^-1']
    jz[:, m, m, z, p] = jz[:, m, m, p, z]
    jz[:, p, z, m, m] = jz[:, m, m, p, z]
    jz[:, z, p, m, m] = jz[:, m, m, p, z]
    
    jz[:, z, m, z, m] = (1/7)*(2/3)**(1/2)*msph['*s_2^-2']+(2/7)*(2/5)**(1/2)*msph['*s_4^-2']
    jz[:, z, m, m, z] = jz[:, z, m, z, m]
    jz[:, m, z, z, m] = jz[:, z, m, z, m]
    jz[:, m, z, m, z] = jz[:, z, m, z, m]
    
    jz[:, z, m, p, m] = (-1/3)*(1/3)**(1/2)*msph['*c_2^-1']+(1/21)*(1/3)**(1/2)*msph['*s_2^-1']+(1/7)*(2/5)**(1/2)*msph['*s_4^-1']
    jz[:, z, m, m, p] = jz[:, z, m, p, m]
    jz[:, p, m, z, m] = jz[:, z, m, p, m]
    jz[:, m, p, z, m] = jz[:, z, m, p, m]
    jz[:, m, z, p, m] = jz[:, z, m, p, m]
    jz[:, m, z, m, p] = jz[:, z, m, p, m]
    jz[:, p, m, m, z] = jz[:, z, m, p, m]
    jz[:, m, p, m, z] = jz[:, z, m, p, m]

    jz[:, z, m, p, z] = (-1/15)*msph['*s_0^0']-(1/21)*msph['*s_2^0']+(4/35)*msph['*s_4^0']
    jz[:, z, m, z, p] = jz[:, z, m, p, z]
    jz[:, p, z, z, m] = jz[:, z, m, p, z]
    jz[:, z, p, z, m] = jz[:, z, m, p, z]
    jz[:, m, z, p, z] = jz[:, z, m, p, z]
    jz[:, m, z, z, p] = jz[:, z, m, p, z]
    jz[:, p, z, m, z] = jz[:, z, m, p, z]
    jz[:, z, p, m, z] = jz[:, z, m, p, z]
    
    jz[:, p, m, p, m] = (1/9)*msph['*c_0^0']+(1/45)*msph['*s_0^0']+(-2/9)*msph['*c_2^0']+(2/63)*msph['*s_2^0']+(2/35)*msph['*s_4^0']
    jz[:, p, m, m, p] = jz[:, p, m, p, m]
    jz[:, m, p, p, m] = jz[:, p, m, p, m]
    jz[:, m, p, m, p] = jz[:, p, m, p, m]

    jz[:, p, m, p, z] = (-1/3)*(1/3)**(1/2)*msph['*c_2^1']+(1/21)*(1/3)**(1/2)*msph['*s_2^1']+(1/7)*(2/5)**(1/2)*msph['*s_4^1']
    jz[:, p, m, z, p] = jz[:, p, m, p, z]
    jz[:, p, z, p, m] = jz[:, p, m, p, z]
    jz[:, z, p, p, m] = jz[:, p, m, p, z]
    jz[:, m, p, p, z] = jz[:, p, m, p, z]
    jz[:, m, p, z, p] = jz[:, p, m, p, z]
    jz[:, p, z, m, p] = jz[:, p, m, p, z]
    jz[:, z, p, m, p] = jz[:, p, m, p, z]
    
    jz[:, p, z, p, z] = (1/7)*(2/3)**(1/2)*msph['*s_2^2']+(2/7)*(2/5)**(1/2)*msph['*s_4^2']
    jz[:, p, z, z, p] = jz[:, p, z, p, z]
    jz[:, z, p, z, p] = jz[:, p, z, p, z]
    jz[:, z, p, p, z] = jz[:, p, z, p, z]
    return jz

def jz2cart(jzC):
    Conv = 1/(2**(1/2))*torch.tensor([[-1, -1j, 0],[0,0,2**(1/2)],[1,-1j,0]], dtype=torch.complex64, device=jzC.device)
    Cdag = Conv.conj().T
    Cdag2 = torch.tensordot(Cdag,Cdag,dims=0)
    Cdag4 = torch.tensordot(Cdag2,Cdag2,dims=0)
    cartC = torch.einsum('ijklmnop,bjlnp->bikmo',Cdag4,jzC)
    return cartC

def cart2voigt(cart: torch.Tensor) -> torch.Tensor:
    """
    Converts a 3x3x3x3 cartesian elastic tensor to its 6x6 voigt form
    """
    voigt = torch.zeros([cart.shape[0], 6, 6], dtype=cart.dtype, device=cart.device)
    x=0
    y=1
    z=2
    voigt[:, 0, 0] = cart[:, x, x, x, x]
    voigt[:, 0, 1] = voigt[:, 1, 0] = cart[:, y, y, x, x]
    voigt[:, 0, 2] = voigt[:, 2, 0] = cart[:, z, z, x, x]
    voigt[:, 0, 3] = voigt[:, 3, 0] = cart[:, y, z, x, x]
    voigt[:, 0, 4] = voigt[:, 4, 0] = cart[:, x, z, x, x]
    voigt[:, 0, 5] = voigt[:, 5, 0] = cart[:, x, y, x, x]
    voigt[:, 1, 1] = cart[:, y, y, y, y]
    voigt[:, 1, 2] = voigt[:, 2, 1] = cart[:, z, z, y, y]
    voigt[:, 1, 3] = voigt[:, 3, 1] = cart[:, y, z, y, y]
    voigt[:, 1, 4] = voigt[:, 4, 1] = cart[:, x, z, y, y]
    voigt[:, 1, 5] = voigt[:, 5, 1] = cart[:, x, y, y, y] 
    voigt[:, 2, 2] = cart[:, z, z, z, z]
    voigt[:, 2, 3] = voigt[:, 3, 2] = cart[:, y, z, z, z]
    voigt[:, 2, 4] = voigt[:, 4, 2] = cart[:, x, z, z, z]
    voigt[:, 2, 5] = voigt[:, 5, 2] = cart[:, x, y, z, z] 
    voigt[:, 3, 3] = cart[:, y, z, y, z]
    voigt[:, 3, 4] = voigt[:, 4, 3] = cart[:, x, z, y, z]
    voigt[:, 3, 5] = voigt[:, 5, 3] = cart[:, x, y, y, z]
    voigt[:, 4, 4] = cart[:, x, z, x, z]
    voigt[:, 4, 5] = voigt[:, 5, 4] = cart[:, x, y, x, z]
    voigt[:, 5, 5] = cart[:, x, y, x, y]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        voigt = voigt.float()
    return voigt


def voigt2cart(voigt: torch.Tensor) -> torch.Tensor:
    """
    Converts a 6x6 voigt elastic tensor to its 3x3x3x3 cartesian form.
    """
    x = 0
    y = 1 
    z = 2
    v11 = voigt[:, 0, 0]
    v12 = voigt[:, 0, 1]
    v13 = voigt[:, 0, 2]
    v14 = voigt[:, 0, 3]
    v15 = voigt[:, 0, 4]
    v16 = voigt[:, 0, 5]
    v22 = voigt[:, 1, 1]
    v23 = voigt[:, 1, 2]
    v24 = voigt[:, 1, 3]
    v25 = voigt[:, 1, 4]
    v26 = voigt[:, 1, 5]
    v33 = voigt[:, 2, 2]
    v34 = voigt[:, 2, 3]
    v35 = voigt[:, 2, 4]
    v36 = voigt[:, 2, 5]
    v44 = voigt[:, 3, 3]
    v45 = voigt[:, 3, 4]
    v46 = voigt[:, 3, 5]
    v55 = voigt[:, 4, 4]
    v56 = voigt[:, 4, 5]
    v66 = voigt[:, 5, 5]

    cart = torch.zeros([voigt.shape[0], 3, 3, 3, 3], dtype=voigt.dtype, device=voigt.device)
    cart[:, x, x, x, x] = v11
    
    cart[:, x, x, y, y] = v12
    cart[:, y, y, x, x] = v12
    
    cart[:, x, x, z, z] = v13
    cart[:, z, z, x, x] = v13
    
    cart[:, x, x, y, z] = v14
    cart[:, x, x, z, y] = v14
    cart[:, z, y, x, x] = v14
    cart[:, y, z, x, x] = v14

    cart[:, x, x, x, z] = v15
    cart[:, x, x, z, x] = v15
    cart[:, x, z, x, x] = v15
    cart[:, z, x, x, x] = v15
    
    cart[:, x, x, x, y] = v16
    cart[:, x, x, y, x] = v16
    cart[:, x, y, x, x] = v16
    cart[:, y, x, x, x] = v16
    
    cart[:, y, y, y, y] = v22

    cart[:, y, y, z, z] = v23
    cart[:, z, z, y, y] = v23

    cart[:, y, y, y, z] = v24
    cart[:, y, y, z, y] = v24
    cart[:, y, z, y, y] = v24
    cart[:, z, y, y, y] = v24
    
    cart[:, y, y, x, z] = v25
    cart[:, y, y, z, x] = v25
    cart[:, x, z, y, y] = v25
    cart[:, z, x, y, y] = v25
    
    cart[:, y, y, x, y] = v26
    cart[:, y, y, y, x] = v26
    cart[:, x, y, y, y] = v26
    cart[:, y, x, y, y] = v26
    
    cart[:, z, z, z, z] = v33
    
    cart[:, z, z, y, z] = v34
    cart[:, z, z, z, y] = v34
    cart[:, y, z, z, z] = v34
    cart[:, z, y, z, z] = v34
    
    cart[:, z, z, x, z] = v35
    cart[:, z, z, z, x] = v35
    cart[:, x, z, z, z] = v35
    cart[:, z, x, z, z] = v35    

    cart[:, z, z, x, y] = v36
    cart[:, z, z, y, x] = v36
    cart[:, x, y, z, z] = v36
    cart[:, y, x, z, z] = v36
    
    cart[:, y, z, y, z] = v44
    cart[:, y, z, z, y] = v44
    cart[:, z, y, y, z] = v44
    cart[:, z, y, z, y] = v44
    
    cart[:, y, z, x, z] = v45
    cart[:, y, z, z, x] = v45
    cart[:, x, z, y, z] = v45
    cart[:, z, x, y, z] = v45
    cart[:, z, y, x, z] = v45
    cart[:, z, y, z, x] = v45
    cart[:, x, z, z, y] = v45
    cart[:, z, x, z, y] = v45

    cart[:, y, z, x, y] = v46
    cart[:, y, z, y, x] = v46
    cart[:, x, y, y, z] = v46
    cart[:, y, x, y, z] = v46
    cart[:, z, y, x, y] = v46
    cart[:, z, y, y, x] = v46
    cart[:, x, y, z, y] = v46
    cart[:, y, x, z, y] = v46
    
    cart[:, x, z, x, z] = v55
    cart[:, x, z, z, x] = v55
    cart[:, z, x, x, z] = v55
    cart[:, z, x, z, x] = v55

    cart[:, x, z, x, y] = v56
    cart[:, x, z, y, x] = v56
    cart[:, x, y, x, z] = v56
    cart[:, y, x, x, z] = v56
    cart[:, z, x, x, y] = v56
    cart[:, z, x, y, x] = v56
    cart[:, x, y, z, x] = v56
    cart[:, y, x, z, x] = v56
    
    cart[:, x, y, x, y] = v66
    cart[:, x, y, y, x] = v66
    cart[:, y, x, y, x] = v66
    cart[:, y, x, x, y] = v66
    
    return cart