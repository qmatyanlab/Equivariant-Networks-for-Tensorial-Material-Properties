import numpy as np
import itertools
import math


def cart2voigt(cart):
    voigt = np.zeros([3,6], dtype=complex)
    x=0
    y=1
    z=2
    voigt[0,0] = cart[x,x,x]
    voigt[0,1] = cart[x,y,y]
    voigt[0,2] = cart[x,z,z]
    voigt[0,3] = cart[x,y,z]
    voigt[0,4] = cart[x,x,z]
    voigt[0,5] = cart[x,x,y]
    
    voigt[1,0] = cart[y,x,x]
    voigt[1,1] = cart[y,y,y]
    voigt[1,2] = cart[y,z,z]
    voigt[1,3] = cart[y,y,z]
    voigt[1,4] = cart[y,x,z]
    voigt[1,5] = cart[y,x,y]
    
    voigt[2,0] = cart[z,x,x]
    voigt[2,1] = cart[z,y,y]
    voigt[2,2] = cart[z,z,z]
    voigt[2,3] = cart[z,y,z]
    voigt[2,4] = cart[z,x,z]
    voigt[2,5] = cart[z,x,y]

    return voigt

def voigt2cart(voigt):
    voigt = np.array(voigt)
    v11 = voigt[0,0]
    v12 = voigt[0,1]
    v13 = voigt[0,2]
    v14 = voigt[0,3]
    v15 = voigt[0,4]
    v16 = voigt[0,5]
    v21 = voigt[1,0]
    v22 = voigt[1,1]
    v23 = voigt[1,2]
    v24 = voigt[1,3]
    v25 = voigt[1,4]
    v26 = voigt[1,5]
    v31 = voigt[2,0]
    v32 = voigt[2,1]
    v33 = voigt[2,2]
    v34 = voigt[2,3]
    v35 = voigt[2,4]
    v36 = voigt[2,5]

    x = 0
    y = 1 
    z = 2
    
    cart = np.zeros([3,3,3], dtype=complex)
    cart[x,x,x] = v11
    cart[y,x,x] = v21
    cart[z,x,x] = v31
    
    cart[x,y,y] = v12
    cart[y,y,y] = v22
    cart[z,y,y] = v32
    
    cart[x,z,z] = v13
    cart[y,z,z] = v23
    cart[z,z,z] = v33
    
    cart[x,y,z] = cart[x,z,y] = v14
    cart[y,y,z] = cart[y,z,y] = v24
    cart[z,y,z] = cart[z,z,y] = v34
        
    cart[x,x,z] = cart[x,z,x] = v15
    cart[y,x,z] = cart[y,z,x] = v25
    cart[z,x,z] = cart[z,z,x] = v35
    
    cart[x,x,y] = cart[x,y,x] = v16
    cart[y,x,y] = cart[y,y,x] = v26
    cart[z,x,y] = cart[z,y,x] = v36
    
    return cart


'''
    Returns symmetric subtensor of cartesian piezoelectric tensor
'''
def r3_symmetrizer(cart):
    S = (1/3)*(cart + np.swapaxes(cart, 0, 1) + np.swapaxes(cart, 0,2))
    return S


'''
    Returns mixed-symmetry subtensor of cartesian piezoelectric tensor
'''

def r3_mixed(cart):
    A = (1/3)*(2*cart - (np.swapaxes(cart, 0, 1) + np.swapaxes(cart, 0,2)))
    return A


'''
    Returns cartesian trace vector of symmetric subspace, note their is only one unique
    trace vector (instead of three) due to it's symmetry
'''

def symmetric_trace_vector(ts):    
    trace_vector = np.einsum('ij,ijk->k',np.eye(3),ts)

    return trace_vector

'''
    Reconstructs rank-3 tensor for trace of symmetric subspace from the cartesian trace vector 
'''

def reconstruct_sym_trace_tensor(trace_vector):

    v1 = np.einsum('ij,k->ijk',np.eye(3),trace_vector)
    v2 = np.einsum('ik,j->ijk',np.eye(3),trace_vector)
    v3 = np.einsum('kj,i->ijk',np.eye(3),trace_vector)

    k=(1/5)*(v1+v2+v3)
    
    return k 

'''
    Returns traceless residue of symmetric subspace, a rank-3 symmetric tensor
'''

def symmetric_residue(ts):
    trace_vector = symmetric_trace_vector(ts)
    k = reconstruct_sym_trace_tensor(trace_vector)
    residue = ts-k
    
    return residue

'''
    Converts a rank-3 tensor from cartesian to jz basis
'''

def r3_cart2jz(S):
    Conv = 1/(2**(1/2))*np.array([[-1, -1j, 0],[0,0,2**(1/2)],[1,-1j,0]])
    C2 = np.tensordot(Conv,Conv,axes=0)
    C3 = np.tensordot(Conv,C2,axes=0)
    jzC = np.einsum('ijklmn,jln->ikm',C3,S)
    return jzC


'''
    Converts a rank-3 tensor from jz basis to cartesian
'''


def r3_jz2cart(jz):
    Conv = 1/(2**(1/2))*np.array([[-1, -1j, 0],[0,0,2**(1/2)],[1,-1j,0]])
    Cdag = Conv.conj().T
    Cdag2 = np.tensordot(Cdag,Cdag,axes=0)
    Cdag3 = np.tensordot(Cdag2,Cdag,axes=0)
    cart = np.einsum('ijklmn,jln->ikm',Cdag3,jz)
    return cart

'''
    Convert TRACELESS Jz basis rank-3 tensor to vector of mochizuki spherical harmonic components
'''

def r3_jz2msph(jzC):
    p=0
    z=1
    m=2
    msph = {
        '*s_3^-3': ((5/2)**(1/2))*jzC[m,m,m],
        '*s_3^-2': ((15/2)**(1/2))*jzC[z,m,m],
        '*s_3^-1': ((3/2)**(1/2))*(2*jzC[z,z,m]+jzC[p,m,m]),
        '*s_3^0': (5/7)*(3*jzC[z,z,z]-jzC[z,p,m]),
        '*s_3^1': ((3/2)**(1/2))*(2*jzC[z,z,p]+jzC[p,p,m]),
        '*s_3^2': ((15/2)**(1/2))*jzC[z,p,p],
        '*s_3^3': ((5/2)**(1/2))*jzC[p,p,p],
        }
    return msph

'''
    Converts TRACELESS rank two tensor from cartesian to mochizuki norm harmonics
'''

def r2_cart2msph(cart):
    x=0
    y=1
    z=2
    msph = {
        '*s_2^-2': (3/2)**(1/2)*(0.5*(cart[x,x] - cart[y,y]) - 1j*cart[x,y]) ,        
        '*s_2^-1': (3/2)**(1/2)*(cart[x,z] - 1j*cart[y,z]) ,        
        '*s_2^0': -0.5*(cart[x,x] + cart[y,y]) + cart[z,z] ,
        '*s_2^1': (3/2)**(1/2)*(-cart[x,z] - 1j*cart[y,z]),
        '*s_2^2': (3/2)**(1/2)*(0.5*(cart[x,x] - cart[y,y]) + 1j*cart[x,y])      
    }
    
    return msph
    

'''
    Convert mochizuki norm to standard harmonic norm (unit over surface integrals)
'''

def r3_msph2sph(msph):
    sph = {
        's_3^-3' : ((7/(4*np.pi))**(1/2))*msph['*s_3^-3'],
        's_3^-2' : ((7/(4*np.pi))**(1/2))*msph['*s_3^-2'],
        's_3^-1' : ((7/(4*np.pi))**(1/2))*msph['*s_3^-1'],
        's_3^0' : ((7/(4*np.pi))**(1/2))*msph['*s_3^0'],
        's_3^1' : ((7/(4*np.pi))**(1/2))*msph['*s_3^1'],
        's_3^2' : ((7/(4*np.pi))**(1/2))*msph['*s_3^2'],
        's_3^3' : ((7/(4*np.pi))**(1/2))*msph['*s_3^3'],
        }

    return sph

def r2_msph2sph(msph):
    sph = {
        's_2^-2' : ((5/(4*np.pi))**(1/2))*msph['*s_2^-2'],
        's_2^-1' : ((5/(4*np.pi))**(1/2))*msph['*s_2^-1'],
        's_2^0' : ((5/(4*np.pi))**(1/2))*msph['*s_2^0'],
        's_2^1' : ((5/(4*np.pi))**(1/2))*msph['*s_2^1'],
        's_2^2' : ((5/(4*np.pi))**(1/2))*msph['*s_2^2']
    }
    
    return sph

'''
    Convert standard norm to mochizuki harmonic norm
'''

def r3_sph2msph(sph):
    msph = {
        '*s_3^-3' : ((7/(4*np.pi))**(-1/2))*sph['s_3^-2'],
        '*s_3^-2' : ((7/(4*np.pi))**(-1/2))*sph['s_3^-2'],
        '*s_3^-1' : ((7/(4*np.pi))**(-1/2))*sph['s_3^-1'],
        '*s_3^0' : ((7/(4*np.pi))**(-1/2))*sph['s_3^0'],
        '*s_3^1' : ((7/(4*np.pi))**(-1/2))*sph['s_3^1'],
        '*s_3^2' : ((7/(4*np.pi))**(-1/2))*sph['s_3^2'],
        '*s_3^3' : ((7/(4*np.pi))**(-1/2))*sph['s_3^2']
    }
    
    return msph

    
def r2_sph2msph(sph):
    msph = {
        '*s_2^-2' : ((5/(4*np.pi))**(-1/2))*sph['s_2^-2'],
        '*s_2^-1' : ((5/(4*np.pi))**(-1/2))*sph['s_2^-1'],
        '*s_2^0' : ((5/(4*np.pi))**(-1/2))*sph['s_2^0'],
        '*s_2^1' : ((5/(4*np.pi))**(-1/2))*sph['s_2^1'],
        '*s_2^2' : ((5/(4*np.pi))**(-1/2))*sph['s_2^2']
    }
    
    return msph



'''
    Converts set of harmonic components into real harmonic components
'''

def r2_sph2realsph(sph):
    realsph = {}    
    realsph['s_2_-2'] = (1j/(2**(1/2)))*(sph['s_2^-2']-sph['s_2^2'])
    realsph['s_2_-1'] = (1j/(2**(1/2)))*(sph['s_2^-1']+sph['s_2^1'])
    realsph['s_2_0'] = sph['s_2^0'] 
    realsph['s_2_1'] = (1/(2**(1/2)))*(sph['s_2^-1']-sph['s_2^1'])    
    realsph['s_2_2'] = (1/(2**(1/2)))*(sph['s_2^-2']+sph['s_2^2'])

    return realsph

def r3_sph2realsph(sph):
    realsph = {}
    
    realsph['s_3_-3'] = (1j/(2**(1/2)))*(sph['s_3^-3']+sph['s_3^3'])
    realsph['s_3_-2'] = (1j/(2**(1/2)))*(sph['s_3^-2']-sph['s_3^2'])
    realsph['s_3_-1'] = (1j/(2**(1/2)))*(sph['s_3^-1']+sph['s_3^1'])
    realsph['s_3_0'] = sph['s_3^0'] 
    realsph['s_3_1'] = (1/(2**(1/2)))*(sph['s_3^-1']-sph['s_3^1'])    
    realsph['s_3_2'] = (1/(2**(1/2)))*(sph['s_3^-2']+sph['s_3^2'])
    realsph['s_3_3'] = (1/(2**(1/2)))*(sph['s_3^-2']+sph['s_3^2'])

    return realsph



'''
    Converts set of real harmonic components into  harmonic components
'''


def r2_realsph2sph(realsph):
    sph = {
    's_2^-2': (2**(-1/2))*(realsph['s_2_2']-1j*realsph['s_2_-2']),
    's_2^-1': (2**(-1/2))*(realsph['s_2_1']-1j*realsph['s_2_-1']),
    's_2^0': realsph['s_2_0'] ,
    's_2^1': -(1/(2**(1/2)))*(realsph['s_2_1']+1j*realsph['s_2_-1']),
    's_2^2': (1/(2**(1/2)))*(realsph['s_2_2']+1j*realsph['s_2_-2']),
    }

    return sph

def r3_realsph2sph(realsph):
    sph = {
    's_3^-3': (2**(-1/2))*(realsph['s_3_3']-1j*realsph['s_3_-3']),
    's_3^-2': (2**(-1/2))*(realsph['s_3_2']-1j*realsph['s_3_-2']),
    's_3^-1': (2**(-1/2))*(realsph['s_3_1']-1j*realsph['s_3_-1']),
    's_3^0': realsph['s_3_0'] ,
    's_3^1': -(1/(2**(1/2)))*(realsph['s_3_1']+1j*realsph['s_3_-1']),
    's_3^2': (1/(2**(1/2)))*(realsph['s_3_2']+1j*realsph['s_3_-2']),
    's_3^3': -(1/(2**(1/2)))*(realsph['s_3_3']+1j*realsph['s_3_-3']),
    }

    return sph

'''
    Converts symmetric, TRACELESS rank-3 cartesian tensor (such as R) to real
    harmonic rank-3 componenet flattened in 1-d np array
'''

def r3_cart2realsphvec(cart):
    jz = r3_cart2jz(cart)
    msph = r3_jz2msph(jz)
    sph = r3_msph2sph(msph)
    realsph = r3_sph2realsph(sph) #### Note here we discard vector components of msph vecs (should be zero)
    r3_real_harmonic_vector = np.array([
        realsph['s_3_-3'],
        realsph['s_3_-2'],
        realsph['s_3_-1'],
        realsph['s_3_0'],
        realsph['s_3_1'],
        realsph['s_3_2'],
        realsph['s_3_3'],
    ])

    return r3_real_harmonic_vector


'''
    Converts rank 2 msph components directly to cartesian tensor
'''

def r2_msph2cart(msph):
    x=0
    y=1
    z=2
    cart = np.array([[0.0,0,0],[0,0,0],[0,0,0]])
    
    cart[x,x] = (1/2)*((2/3)*(-msph['*s_2^0'])+(2/3)**(1/2)*(msph['*s_2^2']+msph['*s_2^-2']))
    cart[y,y] = (1/2)*((2/3)*(-msph['*s_2^0'])-(2/3)**(1/2)*(msph['*s_2^2']+msph['*s_2^-2']))
    cart[z,z] = (1/3)*(2*msph['*s_2^0'])
    cart[x,z] = cart[z,x] = (1/2)*((2/3)**(1/2))*(msph['*s_2^-1']-msph['*s_2^1'])
    cart[y,z] = cart[z,y] = 1j*(1/2)*(2/3)**(1/2)*(msph['*s_2^-1']+msph['*s_2^1'])
    cart[x,y] = cart[y,x] = 1j*(1/2)*(2/3)**(1/2)*(msph['*s_2^-2']-msph['*s_2^2'])
    
    return cart


'''
    Converts mochizuki norm components to jz basis tensor
'''
def r3_msph2jz(msph):
    m=0
    z=1
    p=2
    jz = np.zeros([3,3,3], dtype = complex)
    jz[z,z,z] = (2/5)*msph['*s_3^0']
    jz[p,m,z] = jz[p,z,m] = jz[z,p,m] = jz[m,p,z] = jz[m,z,p] = jz[z,m,p] = (1/5)*msph['*s_3^0']

    jz[p,z,z] = jz[z,p,z] = jz[z,z,p] = ((2/3)**(1/2))*(2/5)*msph['*s_3^1']
    jz[m,z,z] = jz[z,m,z] = jz[z,z,m] = ((2/3)**(1/2))*(2/5)*msph['*s_3^-1']

    jz[m,p,p] = jz[p,p,m] = jz[p,m,p] = ((2/3)**(1/2))*(1/5)*msph['*s_3^1']
    jz[m,p,m] = jz[p,m,m] = jz[m,m,p] = ((2/3)**(1/2))*(1/5)*msph['*s_3^-1']

    jz[z,m,m] = jz[m,z,m] = jz[m,m,z] = ((2/15)**(1/2))*msph['*s_3^-2']
    jz[z,p,p] = jz[p,z,p] = jz[p,p,z] = ((2/15)**(1/2))*msph['*s_3^2']

    jz[m,m,m] = ((2/5)**(1/2))*msph['*s_3^-3']
    jz[p,p,p] = ((2/5)**(1/2))*msph['*s_3^3']

    return jz


'''
    Converts real harmonic components to cartesian tensor
'''

def r3_realsph2cart(realsph):
    sph = r3_realsph2sph(realsph)
    msph = r3_sph2msph(sph)
    jz = r3_msph2jz(msph)
    cart = r3_jz2cart(jz)
    
    return cart


def r2_realsph2cart(realsph):
    sph = r2_realsph2sph(realsph)
    msph = r2_sph2msph(sph)
    
    return r2_msph2cart(msph)


'''
    Converts cartesian tensor directly to real harmonic components
'''

def r2_cart2realsphvec(cart):
    msph = r2_cart2msph(cart)
    sph = r2_msph2sph(msph)
    realsph = r2_sph2realsph(sph)
    realsphvec = np.array([
        realsph['s_2_-2'],
        realsph['s_2_-1'],
        realsph['s_2_0'],
        realsph['s_2_1'],
        realsph['s_2_2']
        ])

    return realsphvec

'''
    Returns cartesian tensor from real spherical component vector (1-d np array)
'''

def r2_realsphvec2cart(realsphvec):
    realsph = {}
    
    realsph['s_2_-2'] = realsphvec[0]  
    realsph['s_2_-1'] = realsphvec[1]  
    realsph['s_2_0'] = realsphvec[2]  
    realsph['s_2_1'] = realsphvec[3]  
    realsph['s_2_2'] = realsphvec[4]  
    
    cart = r2_realsph2cart(realsph)
    
    return cart

def r3_realsphvec2cart(realsphvec):
    realsph = {
        's_3_-3':  realsphvec[0],
        's_3_-2':  realsphvec[1],
        's_3_-1':  realsphvec[2],
        's_3_0':  realsphvec[3],
        's_3_1':  realsphvec[4],
        's_3_2':  realsphvec[5],
        's_3_3':  realsphvec[6],
    }

    cart = r3_realsph2cart(realsph)

    return cart

'''
    Generates a Levi-Civita tensor of input dim
'''

def levi_cevita_tensor(dim):   
    arr=np.zeros(tuple([dim for _ in range(dim)]))
    for x in itertools.permutations(tuple(range(dim))):
        mat = np.zeros((dim, dim), dtype=np.int32)
        for i, j in zip(range(dim), x):
            mat[i, j] = 1
        arr[x]=int(np.linalg.det(mat))
    return arr


'''
    Takes antisymmetric part N of piezotensor and returns trace-vector for N
'''

def mixed_trace_vector(tn):
    trace_vector = np.einsum('jk,ijk->i',np.eye(3),tn)

    return trace_vector


'''
    Takes mixed subtensor's trace vector and returns a reconstructed rank-3 tensor M
'''

def reconstruct_mixed_trace(trace_vector):
    v1 = 2*np.einsum('jk,i->ijk',np.eye(3),trace_vector)
    v2 = -np.einsum('ik,j->ijk',np.eye(3),trace_vector)
    v3 = -np.einsum('ij,k->ijk',np.eye(3),trace_vector)
    
    M = (1/4)*(v1+v2+v3)

    return M

'''
    Returns the traceless residue of N
'''

def mixed_residue(tn):
    trace_vector = mixed_trace_vector(tn)
    M = reconstruct_mixed_trace(trace_vector)
    P = tn - M 
    return P


'''
    Returns a psuedo-tensor for N or P
'''

def mixed_total_psuedotensor(tn):
    eps = levi_cevita_tensor(3)
    b1 = np.einsum('ijk,kmj->im',eps,tn)

    return b1

'''
    Reconstructs P (cartesian) from cartesian psuedotensor
'''

def reconstruct_total_psuedotensor(mixed_residue):
    eps = levi_cevita_tensor(3)
    
    b1 = np.einsum('kpj,pm->kmj',eps,mixed_residue)
    b2 = np.einsum('kpm,pj->kmj',eps,mixed_residue)
    
    P = (1/3)*(b1+b2) ### note here the difference in normalization from the main reference (prev. 1/2)
    
    return P

'''
    Converts cartesian vector to real spherical harmonics
'''

def r1_real(cartesian_vector):
    Conv = 1/(2**(1/2))*np.array([[-1, -1j, 0],[0,0,2**(1/2)],[1,-1j,0]])

    jz = np.einsum('ij,j->i',Conv,cartesian_vector)
    sph = ((3/(4*np.pi))**(1/2))*jz
    realsph = np.array([
        (1j/(2**(1/2)))*(sph[0]+sph[2]),
        sph[1],
        (1/(2**(1/2)))*(sph[0]-sph[2]),
    ])

    return realsph

def r1_unreal(realsph):
    Conv = 1/(2**(1/2))*np.array([[-1, -1j, 0],[0,0,2**(1/2)],[1,-1j,0]])
    Cdag = Conv.conj().T

    sph = np.array([
        (1/(2**(1/2)))*(realsph[2]-1j*realsph[0]),
        realsph[1],
        (-1/(2**(1/2)))*(realsph[2]+1j*realsph[0]),
    ])

    jz = ((3/(4*np.pi))**(-1/2))*sph
    cart = np.einsum('ij,j->i',Cdag,jz)

    return cart



#Troubleshooting functions#
###############################################################################
def reconstruct_trace_psuedotensor(trace_vector):
    eps = levi_cevita_tensor(3)
    B = (1/2)*np.einsum('ijk,k->ij', eps, trace_vector)

    return B

def mixed_psuedotensor_symmetrizer(b1):
    bs = (1/2)*(b1+np.swapaxes(b1,0,1))
    return bs

def mixed_psuedotensor_antisymmetrizer(b1):
    ba = (1/2)*(b1-np.swapaxes(b1,0,1))
    return ba
###############################################################################


'''
    The function you want to use for decomposition: 
        
        Shape([3,6]) Array (Piezoelectric tensor) --> Shape([18]) Array (Harmonic Components)

    Takes as input an array, in voigt notation, as from MP. Returns vector of harmonic components indexed as:

    {
        0: 's_1_-1',        L=1
        1: 's_1_0',         L=1
        2: 's_1_1',         L=1
        3: 's_3_-3',        L=  3
        4: 's_3_-2',        L=  3
        5: 's_3_-1',        L=  3
        6: 's_3_0',         L=  3
        7: 's_3_1',         L=  3
        8: 's_3_2',         L=  3
        9: 's_3_            L=  3
        10: 'n_1_-1',       L=1
        11: 'n_1_0',        L=1
        12: 'n_1_1',        L=1
        13: 'n_2_-2',       L= 2
        14: 'n_2_-1',       L= 2
        15: 'n_2_0',        L= 2
        16: 'n_2_1',        L= 2
        17: 'n_2_2',        L= 2
    }

    where notation indicates _l_m
'''

def real_harmonic_components_piezo(voigt):
    cart = voigt2cart(voigt)
    S = r3_symmetrizer(cart)
    a = symmetric_trace_vector(S)
    R = symmetric_residue(S)
    N = r3_mixed(cart)
    b = mixed_trace_vector(N)
    P = mixed_residue(N)
    B = mixed_total_psuedotensor(P)

    i0_2 = r1_real(a)
    i3_9 = r3_cart2realsphvec(R)
    i10_12 = r1_real(b)
    i13_17 = r2_cart2realsphvec(B)

    harmonic_vec = np.concatenate([i0_2,i3_9,i10_12,i13_17])

    return harmonic_vec

'''
    Reconstruct piezoelectric tensor in voigt notation from real harmonic component vector
'''

def reconstruct_from_harmonic_vec(realsphvec):
    a = realsphvec[0:3]
    a = r1_unreal(a)
    K = reconstruct_sym_trace_tensor(a)

    R = realsphvec[3:10]
    R = r3_realsphvec2cart(R)
    
    b = realsphvec[10:13]
    b = r1_unreal(b)
    Ba = reconstruct_mixed_trace(b)
    
    B = realsphvec[13:18]
    Bs = r2_realsphvec2cart(B)
    Bs = reconstruct_total_psuedotensor(Bs)
    
    piezo_cart = K + R + Ba + Bs
    voigt = cart2voigt(piezo_cart)

    return voigt