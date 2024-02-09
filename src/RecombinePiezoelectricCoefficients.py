import torch, itertools
import warnings

def reconstruct_from_harmonic_vec(realsphvec):
    assert realsphvec.shape[-1] == 18, "Shape mismatch for real spherical vector"
    a = realsphvec[:, 0:3]
    a = r1_unreal(a)
    K = reconstruct_sym_trace_tensor(a)

    R = realsphvec[:, 3:10]
    R = r3_realsphvec2cart(R)
    
    b = realsphvec[:, 10:13]
    b = r1_unreal(b)
    Ba = reconstruct_mixed_trace(b)
    
    B = realsphvec[:, 13:18]
    Bs = r2_realsphvec2cart(B)
    Bs = reconstruct_total_psuedotensor(Bs)
    
    piezo_cart = K + R + Ba + Bs
    voigt = cart2voigt(piezo_cart)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        voigt = voigt.float()
    return voigt

def r1_unreal(realsph):
    Conv = 1/(2**(1/2))*torch.tensor([[-1, -1j, 0],[0,0,2**(1/2)],[1,-1j,0]], dtype=torch.complex64, device=realsph.device)
    Cdag = Conv.conj().T

    sph = torch.stack([
        (1/(2**(1/2)))*(realsph[:, 2]-1j*realsph[:, 0]),
        realsph[:, 1],
        (-1/(2**(1/2)))*(realsph[:, 2]+1j*realsph[:, 0]),
    ], dim=-1)

    jz = ((3/(4*torch.pi))**(-1/2))*sph
    cart = torch.einsum('ij,bj->bi',Cdag,jz)

    return cart

def reconstruct_total_psuedotensor(mixed_residue):
    eps = levi_cevita_tensor(3).to(mixed_residue.device)
    
    b1 = torch.einsum('kpj,bpm->bkmj',eps,mixed_residue)
    b2 = torch.einsum('kpm,bpj->bkmj',eps,mixed_residue)
    
    P = (1/3)*(b1+b2) ### note here the difference in normalization from the main reference (prev. 1/2)
    
    return P

def reconstruct_mixed_trace(trace_vector):
    v1 = 2*torch.einsum('jk,bi->bijk',torch.eye(3, device=trace_vector.device),trace_vector)
    v2 = -torch.einsum('ik,bj->bijk',torch.eye(3, device=trace_vector.device),trace_vector)
    v3 = -torch.einsum('ij,bk->bijk',torch.eye(3, device=trace_vector.device),trace_vector)
    
    M = (1/4)*(v1+v2+v3)

    return M

def cart2voigt(cart):
    voigt = torch.zeros([cart.shape[0],3,6], dtype=torch.complex64, device=cart.device)
    x=0
    y=1
    z=2
    voigt[:, 0,0] = cart[:, x,x,x]
    voigt[:, 0,1] = cart[:, x,y,y]
    voigt[:, 0,2] = cart[:, x,z,z]
    voigt[:, 0,3] = cart[:, x,y,z]
    voigt[:, 0,4] = cart[:, x,x,z]
    voigt[:, 0,5] = cart[:, x,x,y]
    
    voigt[:, 1,0] = cart[:, y,x,x]
    voigt[:, 1,1] = cart[:, y,y,y]
    voigt[:, 1,2] = cart[:, y,z,z]
    voigt[:, 1,3] = cart[:, y,y,z]
    voigt[:, 1,4] = cart[:, y,x,z]
    voigt[:, 1,5] = cart[:, y,x,y]
    
    voigt[:, 2,0] = cart[:, z,x,x]
    voigt[:, 2,1] = cart[:, z,y,y]
    voigt[:, 2,2] = cart[:, z,z,z]
    voigt[:, 2,3] = cart[:, z,y,z]
    voigt[:, 2,4] = cart[:, z,x,z]
    voigt[:, 2,5] = cart[:, z,x,y]

    return voigt

def reconstruct_sym_trace_tensor(trace_vector):

    v1 = torch.einsum('ij,bk->bijk',torch.eye(3, device=trace_vector.device),trace_vector)
    v2 = torch.einsum('ik,bj->bijk',torch.eye(3, device=trace_vector.device),trace_vector)
    v3 = torch.einsum('kj,bi->bijk',torch.eye(3, device=trace_vector.device),trace_vector)

    k=(1/5)*(v1+v2+v3)
    
    return k 

def r3_realsphvec2cart(realsphvec):
    realsph = {
        's_3_-3':  realsphvec[:, 0],
        's_3_-2':  realsphvec[:, 1],
        's_3_-1':  realsphvec[:, 2],
        's_3_0':  realsphvec[:, 3],
        's_3_1':  realsphvec[:, 4],
        's_3_2':  realsphvec[:, 5],
        's_3_3':  realsphvec[:, 6],
    }

    cart = r3_realsph2cart(realsph)

    return cart

def r2_realsphvec2cart(realsphvec):
    realsph = {}
    
    realsph['s_2_-2'] = realsphvec[:, 0]  
    realsph['s_2_-1'] = realsphvec[:, 1]  
    realsph['s_2_0'] = realsphvec[:, 2]  
    realsph['s_2_1'] = realsphvec[:, 3]  
    realsph['s_2_2'] = realsphvec[:, 4]  
    
    cart = r2_realsph2cart(realsph)
    
    return cart


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

def r3_sph2msph(sph):
    msph = {
        '*s_3^-3' : ((7/(4*torch.pi))**(-1/2))*sph['s_3^-3'],
        '*s_3^-2' : ((7/(4*torch.pi))**(-1/2))*sph['s_3^-2'],
        '*s_3^-1' : ((7/(4*torch.pi))**(-1/2))*sph['s_3^-1'],
        '*s_3^0' : ((7/(4*torch.pi))**(-1/2))*sph['s_3^0'],
        '*s_3^1' : ((7/(4*torch.pi))**(-1/2))*sph['s_3^1'],
        '*s_3^2' : ((7/(4*torch.pi))**(-1/2))*sph['s_3^2'],
        '*s_3^3' : ((7/(4*torch.pi))**(-1/2))*sph['s_3^3']
    }
    
    return msph

    
def r2_sph2msph(sph):
    msph = {
        '*s_2^-2' : ((5/(4*torch.pi))**(-1/2))*sph['s_2^-2'],
        '*s_2^-1' : ((5/(4*torch.pi))**(-1/2))*sph['s_2^-1'],
        '*s_2^0' : ((5/(4*torch.pi))**(-1/2))*sph['s_2^0'],
        '*s_2^1' : ((5/(4*torch.pi))**(-1/2))*sph['s_2^1'],
        '*s_2^2' : ((5/(4*torch.pi))**(-1/2))*sph['s_2^2']
    }
    
    return msph

def r3_msph2jz(msph):
    m=2
    z=1
    p=0
    jz = torch.zeros([msph['*s_3^0'].shape[0],3,3,3], dtype=torch.complex64, device=msph['*s_3^0'].device)
    jz[:, z,z,z] = (2/5)*msph['*s_3^0']
    jz[:, p,m,z] = jz[:, p,z,m] = jz[:, z,p,m] = jz[:, m,p,z] = jz[:, m,z,p] = jz[:, z,m,p] = (1/5)*msph['*s_3^0']

    jz[:, p,z,z] = jz[:, z,p,z] = jz[:, z,z,p] = ((2/3)**(1/2))*(2/5)*msph['*s_3^1']
    jz[:, m,z,z] = jz[:, z,m,z] = jz[:, z,z,m] = ((2/3)**(1/2))*(2/5)*msph['*s_3^-1']

    jz[:, m,p,p] = jz[:, p,p,m] = jz[:, p,m,p] = ((2/3)**(1/2))*(1/5)*msph['*s_3^1']
    jz[:, m,p,m] = jz[:, p,m,m] = jz[:, m,m,p] = ((2/3)**(1/2))*(1/5)*msph['*s_3^-1']

    jz[:, z,m,m] = jz[:, m,z,m] = jz[:, m,m,z] = ((2/15)**(1/2))*msph['*s_3^-2']
    jz[:, z,p,p] = jz[:, p,z,p] = jz[:, p,p,z] = ((2/15)**(1/2))*msph['*s_3^2']

    jz[:, m,m,m] = ((2/5)**(1/2))*msph['*s_3^-3']
    jz[:, p,p,p] = ((2/5)**(1/2))*msph['*s_3^3']

    return jz

def r3_jz2cart(jz):
    Conv = 1/(2**(1/2))*torch.tensor([[-1, -1j, 0],[0,0,2**(1/2)],[1,-1j,0]], dtype=torch.complex64, device=jz.device)
    Cdag = Conv.conj().T
    Cdag2 = torch.tensordot(Cdag,Cdag,dims=0)
    Cdag3 = torch.tensordot(Cdag2,Cdag,dims=0)
    cart = torch.einsum('ijklmn,bjln->bikm',Cdag3,jz)
    return cart

def r2_msph2cart(msph):
    x=0
    y=1
    z=2
    cart = torch.zeros([msph['*s_2^-2'].shape[0],3,3], dtype=torch.complex64, device=msph['*s_2^-2'].device)
    
    cart[:, x,x] = (1/2)*((2/3)*(-msph['*s_2^0'])+(2/3)**(1/2)*(msph['*s_2^2']+msph['*s_2^-2']))
    cart[:, y,y] = (1/2)*((2/3)*(-msph['*s_2^0'])-(2/3)**(1/2)*(msph['*s_2^2']+msph['*s_2^-2']))
    cart[:, z,z] = (1/3)*(2*msph['*s_2^0'])
    cart[:, x,z] = cart[:, z,x] = (1/2)*((2/3)**(1/2))*(msph['*s_2^-1']-msph['*s_2^1'])
    cart[:, y,z] = cart[:, z,y] = 1j*(1/2)*(2/3)**(1/2)*(msph['*s_2^-1']+msph['*s_2^1'])
    cart[:, x,y] = cart[:, y,x] = 1j*(1/2)*(2/3)**(1/2)*(msph['*s_2^-2']-msph['*s_2^2'])
    
    return cart

def levi_cevita_tensor(dim):   
    arr=torch.zeros([dim for _ in range(dim)], dtype=torch.complex64)
    for x in itertools.permutations(range(dim)):
        mat = torch.zeros((dim, dim), dtype=torch.float32)
        for i, j in zip(range(dim), x):
            mat[i, j] = 1
        arr[x]=int(torch.linalg.det(mat))
    return arr