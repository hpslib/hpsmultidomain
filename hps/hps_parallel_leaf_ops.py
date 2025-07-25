import torch  # Used for tensor operations with GPU support
import numpy as np  # For numerical operations, especially those related to array manipulation

torch.set_default_dtype(torch.double)  # Ensure all torch tensors are double precision for accuracy

# Utility function to find the largest divisor of 'n' that is smaller or equal to 'div'
def get_nearest_div(n, div):
    """
    Finds the largest divisor of n that is less than or equal to div.
    
    Parameters:
    - n (int): The number to find the divisor for.
    - div (int): The maximum allowed divisor.
    
    Returns:
    - int: The largest divisor of n that does not exceed div.
    """
    while (np.mod(n, div) > 0):
        div -= 1
    return div

# Function to assemble local blocks of the global matrix for a 2D domain
def get_Aloc_2d(p, xxloc, Ds, pdo, box_start, box_end, device):
    """
    Assembles local blocks of the global matrix representing the differential operator in a 2D domain.
    
    Parameters:
    - p (int): The polynomial degree for spectral methods or the discretization parameter for FD.
    - xxloc (tensor): The locations of the grid points.
    - Ds (list of tensors): The differential operators (e.g., derivatives) in matrix form.
    - pdo (object): An object containing the functions for the coefficients of the PDE.
    - box_start, box_end (int): Indices defining the range of boxes to process.
    - device (torch.device): The computation device (CPU or GPU).
    
    Returns:
    - Aloc (tensor): A tensor containing the assembled local blocks of the global matrix.
    """
    nboxes = box_end - box_start  # Number of boxes to process
    Aloc = torch.zeros(nboxes, p**2, p**2, device=device)  # Initialize the tensor for local blocks
    xx_flat = xxloc[box_start:box_end].reshape(nboxes*p**2, 2)  # Flatten the grid points for the given range

    # Accumulate the contributions of each coefficient to the local blocks
    Aloc_acc(p, 2, nboxes, xx_flat, Aloc, pdo.c11, Ds[0], c=-1.)
    Aloc_acc(p, 2, nboxes, xx_flat, Aloc, pdo.c22, Ds[1], c=-1.)
    if pdo.c12 is not None:
        Aloc_acc(p, 2, nboxes, xx_flat, Aloc, pdo.c12, Ds[2], c=-1.)
    if pdo.c1 is not None:
        Aloc_acc(p, 2, nboxes, xx_flat, Aloc, pdo.c1, Ds[3])
    if pdo.c2 is not None:
        Aloc_acc(p, 2, nboxes, xx_flat, Aloc, pdo.c2, Ds[4])
    if pdo.c is not None:
        I = torch.eye(p**2, device=device)
        Aloc_acc(p, 2, nboxes, xx_flat, Aloc, pdo.c, I)
    return Aloc

# Function to assemble local blocks of the global matrix for a 3D domain
def get_Aloc_3d(p, xxloc, Ds, pdo, box_start, box_end, device):
    """
    Assembles local blocks of the global matrix representing the differential operator in a 2D domain.
    
    Parameters:
    - p (int): The polynomial degree for spectral methods or the discretization parameter for FD.
    - xxloc (tensor): The locations of the grid points.
    - Ds (list of tensors): The differential operators (e.g., derivatives) in matrix form.
    - pdo (object): An object containing the functions for the coefficients of the PDE.
    - box_start, box_end (int): Indices defining the range of boxes to process.
    - device (torch.device): The computation device (CPU or GPU).
    
    Returns:
    - Aloc (tensor): A tensor containing the assembled local blocks of the global matrix.
    """
    nboxes = box_end - box_start  # Number of boxes to process
    Aloc = torch.zeros(nboxes, p**3, p**3, device=device)  # Initialize the tensor for local blocks
    xx_flat = xxloc[box_start:box_end].reshape(nboxes*p**3, 3)  # Flatten the grid points for the given range

    # Accumulate the contributions of each coefficient to the local blocks
    Aloc_acc(p, 3, nboxes, xx_flat, Aloc, pdo.c11, Ds[0], c=-1.)
    Aloc_acc(p, 3, nboxes, xx_flat, Aloc, pdo.c22, Ds[1], c=-1.)
    Aloc_acc(p, 3, nboxes, xx_flat, Aloc, pdo.c33, Ds[2], c=-1.)
    if pdo.c12 is not None:
        Aloc_acc(p, 3, nboxes, xx_flat, Aloc, pdo.c12, Ds[3], c=-1.)
    if pdo.c13 is not None:
        Aloc_acc(p, 3, nboxes, xx_flat, Aloc, pdo.c13, Ds[4], c=-1.)
    if pdo.c23 is not None:
        Aloc_acc(p, 3, nboxes, xx_flat, Aloc, pdo.c23, Ds[5], c=-1.)
    if pdo.c1 is not None:
        Aloc_acc(p, 3, nboxes, xx_flat, Aloc, pdo.c1, Ds[6])
    if pdo.c2 is not None:
        Aloc_acc(p, 3, nboxes, xx_flat, Aloc, pdo.c2, Ds[7])
    if pdo.c3 is not None:
        Aloc_acc(p, 3, nboxes, xx_flat, Aloc, pdo.c3, Ds[8])
    if pdo.c is not None:
        I = torch.eye(p**3, device=device)
        Aloc_acc(p, 3, nboxes, xx_flat, Aloc, pdo.c, I)
    return Aloc

# Helper function to accumulate the contribution of each coefficient function to the local blocks
def Aloc_acc(p, d, nboxes, xx_flat, Aloc, func, D, c=1.):
    """
    Accumulates the contribution of a coefficient function to the local blocks of the global matrix.
    
    Parameters:
    - p, d (int): The polynomial degree and the dimensionality of the problem, respectively.
    - nboxes (int): The number of boxes to process.
    - xx_flat (tensor): The flattened locations of the grid points for the given range.
    - Aloc (tensor): The tensor for the local blocks being assembled.
    - func (callable): The coefficient function to be applied.
    - D (tensor): The differential operator matrix related to the coefficient function.
    - c (float): A scaling factor for the coefficient function (default is 1).
    """
    size_f    = p**d
    f_vals    = func(xx_flat).reshape(nboxes,size_f)
    f_vals    = f_vals[:,:,None]
    f_vals   *= c
    Aloc     += f_vals * D.unsqueeze(0)

    
def form_DtNs(p,d,xxloc,Nx,Nxc,Jx,Jc,Jxreo,Jxun,Ds,Intmap,Intmap_rev,Intmap_unq,pdo,
          box_start,box_end,device,mode,interpolate,data,ff_body_func,ff_body_vec,uu_true):
    args = p,xxloc,Ds,pdo,box_start,box_end
    """
    Do (potentially GPU) operations on PyTorch tensors.
    Modes correspond to different operations: computing DtNs, leaf solves, or redcuing the condensed RHS.
    All tensors should be small enough to fit on the GPU (if using one)
    """

    # We need Aloc:
    if (d == 2):
        Aloc = get_Aloc_2d(*args,device)
    else:
        Aloc = get_Aloc_3d(*args,device)
    Acc = Aloc[:,Jc[:,None],Jc]

    if ff_body_vec is not None:
        ff_body_vec = ff_body_vec.to(device)

    # This one doesn't require Aloc
    if (mode == 'reduce_body'):
        nrhs = 1 # Assuming 1 RHS for now
        f_body  = torch.zeros((box_end-box_start),(p-2)**d,nrhs,device=device)
        if ff_body_func is not None:
            xx_flat = xxloc[box_start:box_end,Jc].reshape((box_end-box_start)*(p-2)**d,d)
            tmp = ff_body_func(xx_flat)
            f_body += tmp.reshape(box_end-box_start,(p-2)**d,nrhs)
        if ff_body_vec is not None:
            f_body_vec_part = ff_body_vec[box_start*p**d:box_end*p**d].reshape(box_end-box_start,p**d,nrhs)
            f_body += f_body_vec_part[:,Jc,:]

        if interpolate == False:
            return -Nx[:,Jc].unsqueeze(0) @ torch.linalg.solve(Acc,f_body)
        else:
            return -Intmap_rev.unsqueeze(0) @ Nxc[:,Jc].unsqueeze(0) @ torch.linalg.solve(Acc,f_body)

    elif (mode == 'build'):
        nrhs = data.shape[-1]
        
        # Slightly different operations depending on if we use Gaussian interpolation or not:
        if interpolate == False:
            S_tmp  = -torch.linalg.solve(Acc, Aloc[:,Jc[:,None],Jx])
            Irep   = torch.eye(Jx.shape[0],device=device).unsqueeze(0).repeat(box_end-box_start,1,1)
            S_full = torch.concat((S_tmp,Irep),dim=1)

            Jtot = torch.hstack((Jc,Jx))
            DtN  = Nx[...,Jtot].unsqueeze(0) @ S_full

        else:
            S_tmp   = -torch.linalg.solve(Acc, Aloc[:,Jc[:,None],Jxun])
            Intmap_repeat = Intmap_unq.unsqueeze(0).repeat(box_end-box_start,1,1)
            S_full        = torch.concat((S_tmp @ Intmap_unq.unsqueeze(0),Intmap_repeat),dim=1) # Applying interpolation to both identity and S
            
            Jtot = torch.hstack((Jc,Jxun))
            DtN  = Nxc[:,Jtot].unsqueeze(0) @ S_full
            DtN  = Intmap_rev.unsqueeze(0) @ DtN
        return DtN
    
    # Solve for the subdomain interiors:
    elif (mode == 'solve'):

        # First we may need to compute the RHS if it has a body load:
        nrhs   = data.shape[-1]
        f_body = torch.zeros(box_end-box_start,Jc.shape[0],nrhs,device=device)
        if (ff_body_func is not None):
            xx_flat = xxloc[box_start:box_end,Jc].reshape((box_end-box_start)*(p-2)**d,d)
            f_body += ff_body_func(xx_flat).reshape(box_end-box_start,(p-2)**d,nrhs)
        if (ff_body_vec is not None):
            f_body_vec_part = ff_body_vec[box_start*p**d:box_end*p**d].reshape(box_end-box_start,p**d,nrhs)
            f_body         += f_body_vec_part[:,Jc]
        

        uu_sol = torch.zeros(box_end-box_start,p**d,2*nrhs,device=device)
        
        # We use different indexing schemes for dropped corners and Gaussian:
        if interpolate == False:
            uu_sol[:,Jx,:nrhs] = data[box_start:box_end]
            uu_sol[:,Jc,:nrhs] = -Aloc[:,Jc[:,None],Jx] @ data[box_start:box_end]
        else:
            # Need to make this Jxunique:
            uu_sol[:,Jxun,:nrhs] = Intmap_unq.unsqueeze(0) @ data[box_start:box_end]
            uu_sol[:,Jc,:nrhs]   = -Aloc[:,Jc[:,None],Jxun] @ uu_sol[:,Jxun,:nrhs]

        uu_sol[:,Jc,:nrhs]  += f_body    
        uu_sol[:,Jc,:nrhs]   = torch.linalg.solve(Acc, uu_sol[:,Jc,:nrhs])
            
        # calculate residual if there is a true solution available
        if uu_true is None:
            uu_sol[:,Jc,nrhs:] = Aloc[:,Jc] @ uu_sol[...,:nrhs] - f_body
        else:
            uu_sol[:,Jc,nrhs:] = Aloc[:,Jc] @ uu_true[box_start:box_end] - f_body
        return uu_sol
    
def get_DtN_chunksize(p,d,device,mode):
    """
    Computing leaf solves or DtNs requires significant memory overhead beyond the cost of storing
    the DtNs / solutions themselves. Here the free space on the GPU is computed to determine how many 
    can be done at once without erroring.
    """
    q = p-2
    if (device == torch.device('cuda')):
        r = torch.cuda.memory_reserved(0)
        a = torch.cuda.memory_allocated(0)
        f = r-a # in bytes
        #print(f"Available memory for next chunk: {r} - {a} = {f}")
    else:
        f = 10e9 # 10 GB in bytes
    chunk_max = int(f / (4*p**(2*d) * 8)) # 8 bytes in 64 bits memory
    if d == 3:
        chunk_max = int(f / ((p**6 + q**6 + 12*q**5 + 72*q**4) * 8)) # 8 bytes in 64 bits memory
        if mode=="solve":
            # Need a little extra overhead for Aloc in this case
            chunk_max = int(f / ((2*p**6 + q**6 + 12*q**5 + 72*q**4) * 8))
        if p >= 16:
            chunk_max = 1
    return np.max([chunk_max, 1])


def get_DtNs_helper(p,q,d,xxloc,Nx,Nxc,Jx,Jc,Jxreo,Jxun,Ds,Intmap,Intmap_rev,Intmap_unq,pdo,\
                    box_start,box_end,chunk_init,device,mode,interpolate,data,ff_body_func,ff_body_vec,uu_true):
    """
    Handles the batch scheduling for computing DtNs and leaf solves - designates how many
    are computed at once, and transfers results to CPU as needed.
    """
    nboxes = box_end - box_start
    size_face = q**(d-1)
    if (mode == 'build'):
        DtNs = torch.zeros(nboxes,2*d*size_face,2*d*size_face,device=device)
    elif (mode == 'solve'):
        DtNs = torch.zeros(nboxes,p**d,2*data.shape[-1],device=device)
    elif (mode == 'reduce_body'):
        DtNs = torch.zeros(nboxes,2*d*size_face,1,device=device)
    #print("Built zero arrays in helper")
    chunk_size = chunk_init
    args = p,d,xxloc,Nx,Nxc,Jx,Jc,Jxreo,Jxun,Ds,Intmap,Intmap_rev,Intmap_unq,pdo
    chunk_list = torch.zeros(int(nboxes/chunk_init)+100,device=device).int(); 
    box_curr = 0; nchunks = 0

    while(box_curr < nboxes):
        b1 = box_curr + box_start
        b2 = np.min([box_end, b1 + chunk_size])

        tmp = form_DtNs(*args,b1,b2,device,mode,interpolate,data,ff_body_func,ff_body_vec,uu_true)
        
        DtNs[box_curr:box_curr + chunk_size] = tmp
        box_curr += chunk_size

        chunk_size = get_DtN_chunksize(p,d,device,mode) #np.max([get_DtN_chunksize(p,d,device),chunk_init])
        chunk_list[nchunks] = b2-b1
        nchunks += 1
    return DtNs.cpu(),chunk_list[:nchunks]
