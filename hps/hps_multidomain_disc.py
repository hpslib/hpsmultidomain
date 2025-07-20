import torch  # For tensor operations and GPU support
import numpy as np  # Numerical operations, especially for non-tensor computations
import scipy.sparse as sp  # Sparse matrix operations for efficient memory usage
from time import time  # Tracking execution times

# Importing HPS discretization and parallel leaf operations modules
import hps_leaf_disc as hps_disc  
import hps_parallel_leaf_ops as leaf_ops  

# Utility function to create batched meshgrid for tensor operations
def batched_meshgrid(b, npoints, I, J):
    """
    Creates a batched version of the meshgrid function, useful for vectorized operations over batches.
    
    Parameters:
    - b (int): Batch size.
    - npoints (int): Number of points along each axis in the grid.
    - I, J (torch.Tensor): Tensors representing the indices for meshgrid generation, both of shape (b, npoints).
    
    Returns:
    - zz1, zz2 (torch.Tensor): Two tensors representing the meshgrid coordinates, each of shape (b, npoints, npoints).
    """
    # Ensures input shapes are as expected
    assert I.shape[0] == b and I.shape[1] == npoints
    assert J.shape[0] == b and J.shape[1] == npoints
    # Generates the batched meshgrid
    zz1 = torch.repeat_interleave(I, npoints).reshape(b, npoints, npoints)
    zz2 = torch.repeat_interleave(J, npoints).reshape(b, -1, npoints)
    zz2 = torch.transpose(zz2, -1, -2)  # Correcting the orientation
    return zz1, zz2

# HPS Multidomain class for handling multidomain discretizations and solutions
class HPS_Multidomain:
    
    def __init__(self, pdo, domain, a, p, d, periodic_bc=False):
        """
        Initializes the HPS multidomain solver with domain information and discretization parameters.
        
        Parameters:
        - pdo: An object representing the partial differential operator.
        - domain (torch.Tensor): The computational domain represented as a tensor.
        - a (float): Characteristic length scale for the domain.
        - p (int): Polynomial degree for spectral methods or discretization parameter.
        - d (int): Spatial dimension of the problem and corresponding discretization
        """
        self.pdo    = pdo
        self.domain = domain
        self.p      = p
        self.a      = a
        self.d      = d

        self.periodic_bc = periodic_bc

        # For interpolation:
        self.interpolate = True
        self.q = self.p - 2

        if d==2:
            if (pdo.c12 is None):
                self.interpolate = False
                self.q = self.p - 2
        else: # d==3
            if (pdo.c12 is None) and (pdo.c13 is None) and (pdo.c23 is None):
                self.interpolate = False
                self.q = self.p - 2
        
        n = (self.domain[:,1] - self.domain[:,0]) / (2*self.a)
        n = torch.round(n).int()
        nboxes = torch.prod(n)
        self.n = n; self.nboxes = nboxes
        self.H = hps_disc.HPS_Disc(a,p,self.q,d)
        self.hmin = self.H.hmin

        
        print("Boxes", self.n)
        print("Total boxes", self.nboxes)
        
        Dtmp  = self.H.Ds
        Ds = 0
        if d==2:
            Ds = torch.stack((torch.tensor(Dtmp.D11), torch.tensor(Dtmp.D22),\
                              torch.tensor(Dtmp.D12),\
                              torch.tensor(Dtmp.D1), torch.tensor(Dtmp.D2)))
        else:
            Ds = torch.stack((torch.tensor(Dtmp.D11), torch.tensor(Dtmp.D22), torch.tensor(Dtmp.D33),\
                              torch.tensor(Dtmp.D12), torch.tensor(Dtmp.D13), torch.tensor(Dtmp.D23),\
                              torch.tensor(Dtmp.D1), torch.tensor(Dtmp.D2), torch.tensor(Dtmp.D3)))
        self.H.Ds = Ds 
        
        Jx = torch.tensor(self.H.JJ.Jx)
        Jxreorder = torch.tensor(self.H.JJ.Jxreorder)
        
        self.grid_xx  = self.get_grid()
        self.grid_ext = self.grid_xx[:,Jxreorder,:].flatten(start_dim=0,end_dim=-2)
        self.gauss_xx = self.get_gaussian_nodes()
        self.gauss_xx = self.gauss_xx.flatten(start_dim=0,end_dim=-2)
        self.I_unique, self.I_copy1, self.I_copy2 = self.get_unique_inds()

        # We want xx_ext to be based on Gaussian nodes unless there are no mixed terms:
        self.xx_ext = self.gauss_xx
        if self.interpolate == False:
            self.xx_ext = self.grid_xx[:,Jx,:].flatten(start_dim=0,end_dim=-2)
        self.xx_active = self.xx_ext[self.I_unique,:]
        
        self.xx_tot = self.grid_xx.flatten(start_dim=0,end_dim=-2)
    
    def sparse_mat(self, device, verbose=False):
        """
        Constructs a sparse matrix representation of the problem on the specified device.
        
        Parameters:
        - device (torch.device): The device (CPU or GPU) for computation.
        - verbose (bool): Flag to enable detailed printouts for debugging or monitoring.
        
        Returns:
        - sp_mat: Sparse matrix representation of the HPS operator.
        - t_dict: Dictionary containing timing information for different parts of the matrix assembly process.
        """
        #print("Building DtNs")
        tic = time()
        DtN_loc = self.get_DtNs(device,'build')
        toc_DtN = time() - tic
        """
        from torch.profiler import profile, record_function, ProfilerActivity
        activities = [ProfilerActivity.CPU]
        if torch.cuda.is_available():
            device_string = 'cuda'
            activities += [ProfilerActivity.CUDA]
        else:
            device_string = 'cpu'
        sort_by_keyword = device_string + "_time_total"
        with profile(activities=activities, record_shapes=True) as prof:
            with record_function("get_DtNs"):
                self.get_DtNs(device,'build')
        print(prof.key_averages(group_by_input_shape=False).table(sort_by=sort_by_keyword, row_limit=12))
        """
        
        tic = time()
        sp_mat = sp.block_diag(DtN_loc)
        sp_mat = sp_mat.tocsr()
        toc_csr_scipy = time() - tic

        print("Assembled sparse matrix")

        t_dict = dict()
        t_dict['toc_DtN'] = toc_DtN
        t_dict['toc_sparse'] = toc_csr_scipy
        return sp_mat,t_dict
    
    def get_grid(self):
        """
        Generates the computational grid based on the discretization parameters and domain geometry.
        
        Returns:
        - xx (torch.Tensor): Tensor representing the grid points in the computational domain.
        """
        zz = torch.tensor(self.H.zz.T)

        n = self.n
        xx = torch.zeros(self.nboxes, self.p**self.d, self.d)
        for i in range(n[0]):
            for j in range(n[1]):
                if self.d==2:
                    box   = i*n[1] + j
                    zzloc = zz.clone()
                    zzloc[:,0] += self.a[0] + 2*self.a[0]*i + self.domain[0,0]
                    zzloc[:,1] += self.a[1] + 2*self.a[1]*j + self.domain[1,0]
                    xx[box,:,:] = zzloc
                else:
                    for k in range(n[2]):
                        box   = i*n[1]*n[2] + j*n[2] + k
                        zzloc = zz.clone()
                        zzloc[:,0] += self.a[0] + 2*self.a[0]*i + self.domain[0,0]
                        zzloc[:,1] += self.a[1] + 2*self.a[1]*j + self.domain[1,0]
                        zzloc[:,2] += self.a[2] + 2*self.a[2]*k + self.domain[2,0]
                        xx[box,:,:] = zzloc

        return xx

    def get_gaussian_nodes(self):
        """
        Generates the computational box exteriors based on the discretization parameters and domain geometry.
        
        Returns:
        - xxG (torch.Tensor): Tensor representing the Gaussian grid points of the box surfaces in the computational domain.
        """
        zzG = torch.tensor(self.H.zzG)
        #print(zzG.shape)
        n   = self.n
        xxG = torch.zeros(self.nboxes, 2*self.d*self.q**(self.d-1), self.d)
        for i in range(n[0]):
            for j in range(n[1]):
                if self.d==2:
                    box   = i*n[1] + j
                    zzloc = zzG.clone()
                    zzloc[:,0] += self.a[0] + 2*self.a[0]*i + self.domain[0,0]
                    zzloc[:,1] += self.a[1] + 2*self.a[1]*j + self.domain[1,0]
                    xxG[box,:,:] = zzloc
                else:
                    for k in range(n[2]):
                        box   = i*n[1]*n[2] + j*n[2] + k
                        zzloc = zzG.clone()
                        zzloc[:,0] += self.a[0] + 2*self.a[0]*i + self.domain[0,0]
                        zzloc[:,1] += self.a[1] + 2*self.a[1]*j + self.domain[1,0]
                        zzloc[:,2] += self.a[2] + 2*self.a[2]*k + self.domain[2,0]
                        xxG[box,:,:] = zzloc

        return xxG
    
    def get_unique_inds(self):
        """
        Identifies unique and duplicated indices for handling boundary conditions and overlaps between subdomains.
        
        Returns:
        - I_unique, I_copy1, I_copy2 (torch.Tensor): Tensors representing unique and duplicated grid indices.
        """
        if self.d==2:
            # Assuming gaussian nodes with pxp nodes total
            # FOR NOW we're assuming Chebyshev
            size_face = self.q
            n0,n1  = self.n

            box_ind = torch.arange(n0*n1*4*size_face).reshape(n0,n1,4*size_face)

            # Keep order in mind: L R D U B F (n0 is LR, n1 is DU, n2 is BF)
            # Unique: 1 copy of every boundary in the model. This will be:
            # ALL down, left, and back faces
            # Right face for the rightmost boxes (on n0) UNLESS we have periodic BC
            # Up face for the upmost boxes (on n1)
            I_unique = box_ind.clone()
            if self.periodic_bc:
                I_unique[:,:,size_face:2*size_face] = -1 # Eliminate right edges
            else:
                I_unique[:-1,:,size_face:2*size_face] = -1 # Eliminate right edges except rightmost

            I_unique[:,:-1,3*size_face:4*size_face] = -1 # Eliminate up edges except upmost
            I_unique = I_unique.flatten()
            I_unique = I_unique[I_unique > -1]

            # For copy 1, we need to eliminate edges that make up domain boundary. This is just:
            # Left faces for all but leftmost boxes (UNLESS we have a periodic domain on the L/R boundary)
            # Down faces for all but downmost boxes
            indices_ru = np.hstack((np.arange(size_face,2*size_face),np.arange(3*size_face,4*size_face)))
            I_copy1 = box_ind.clone()
            I_copy1[:,:,indices_ru]              = -1 # Eliminate all right and up faces
            if not self.periodic_bc:
                I_copy1[0,:,:size_face]          = -1 # Eliminate left faces on left edge if they aren't periodic
            I_copy1[:,0,2*size_face:3*size_face] = -1 # Eliminate down faces on down edge
            I_copy1 = I_copy1.flatten()
            I_copy1 = I_copy1[I_copy1 > -1]
            #print("I_copy1 shape is " + str(I_copy1.shape))
            #print(I_copy1)

            # For copy 2, we need to match relative indexing of copy 1 to easily copy from one to the other.
            # We'll do this by copying the correct copy 2 indices to their relative points in copy1,
            # then mimic the eliminations we did in copy1
            I_copy2 = box_ind.clone()

            # Every down index is equal to the up of the preceding box
            I_copy2[:,1:,2*size_face:3*size_face] = I_copy2[:,:-1,3*size_face:4*size_face]
            # Every left index is equal to the right of the preceding box
            I_copy2[1:,:,:size_face] = I_copy2[:-1,:,size_face:2*size_face]

            # SPECIAL CASE: if periodic, we need the leftmost domain faces to equal the rightmost:
            if self.periodic_bc:
                I_copy2[0,:,:size_face] = I_copy2[-1,:,size_face:2*size_face]


            I_copy2[:,:,indices_ru]              = -1 # Eliminate all right, up, and front faces
            if not self.periodic_bc:
                I_copy2[0,:,:size_face]          = -1 # Eliminate left faces on left edge if they aren't periodic
            I_copy2[:,0,2*size_face:3*size_face] = -1 # Eliminate down faces on down edge
            I_copy2 = I_copy2.flatten()
            I_copy2 = I_copy2[I_copy2 > -1]
        else:
            # Assuming gaussian nodes with pxp nodes total
            # FOR NOW we're assuming Chebyshev
            size_face = self.q**2
            n0,n1,n2  = self.n

            box_ind = torch.arange(n0*n1*n2*6*size_face).reshape(n0,n1,n2,6*size_face)

            # Keep order in mind: L R D U B F (n0 is LR, n1 is DU, n2 is BF)
            # Unique: 1 copy of every boundary in the model. This will be:
            # ALL down, left, and back faces
            # Right face for the rightmost boxes (on n0) UNLESS we have periodic BC
            # Up face for the upmost boxes (on n1)
            # Front face for the frontmost boxes (on n2)
            I_unique = box_ind.clone()
            if self.periodic_bc:
                I_unique[:,:,:,size_face:2*size_face] = -1 # Eliminate right edges
            else:
                I_unique[:-1,:,:,size_face:2*size_face] = -1 # Eliminate right edges except rightmost

            I_unique[:,:-1,:,3*size_face:4*size_face] = -1 # Eliminate up edges except upmost
            I_unique[:,:,:-1,5*size_face:] = -1 # Eliminate front edges except frontmost
            I_unique = I_unique.flatten()
            I_unique = I_unique[I_unique > -1]

            # For copy 1, we need to eliminate edges that make up domain boundary. This is just:
            # Left faces for all but leftmost boxes (UNLESS we have a periodic domain on the L/R boundary)
            # Down faces for all but downmost boxes
            # Back faces for all but backmost boxes
            indices_ruf = np.hstack((np.arange(size_face,2*size_face),np.arange(3*size_face,4*size_face),np.arange(5*size_face,6*size_face)))
            I_copy1 = box_ind.clone()
            I_copy1[:,:,:,indices_ruf]             = -1 # Eliminate all right, up, and front faces
            if not self.periodic_bc:
                I_copy1[0,:,:,:size_face]          = -1 # Eliminate left faces on left edge if they aren't periodic
            I_copy1[:,0,:,2*size_face:3*size_face] = -1 # Eliminate down faces on down edge
            I_copy1[:,:,0,4*size_face:5*size_face] = -1 # Eliminate back faces on back edge
            I_copy1 = I_copy1.flatten()
            I_copy1 = I_copy1[I_copy1 > -1]
            #print("I_copy1 shape is " + str(I_copy1.shape))
            #print(I_copy1)

            # For copy 2, we need to match relative indexing of copy 1 to easily copy from one to the other.
            # We'll do this by copying the correct copy 2 indices to their relative points in copy1,
            # then mimic the eliminations we did in copy1
            I_copy2 = box_ind.clone()

            # Every back index is equal to the front of the preceding box
            I_copy2[:,:,1:,4*size_face:5*size_face] = I_copy2[:,:,:-1,5*size_face:]
            # Every down index is equal to the up of the preceding box
            I_copy2[:,1:,:,2*size_face:3*size_face] = I_copy2[:,:-1,:,3*size_face:4*size_face]
            # Every left index is equal to the right of the preceding box
            I_copy2[1:,:,:,:size_face] = I_copy2[:-1,:,:,size_face:2*size_face]

            # SPECIAL CASE: if periodic, we need the leftmost domain faces to equal the rightmost:
            if self.periodic_bc:
                I_copy2[0,:,:,:size_face] = I_copy2[-1,:,:,size_face:2*size_face]


            I_copy2[:,:,:,indices_ruf]             = -1 # Eliminate all right, up, and front faces
            if not self.periodic_bc:
                I_copy2[0,:,:,:size_face]          = -1 # Eliminate left faces on left edge if they aren't periodic
            I_copy2[:,0,:,2*size_face:3*size_face] = -1 # Eliminate down faces on down edge
            I_copy2[:,:,0,4*size_face:5*size_face] = -1 # Eliminate back faces on back edge
            I_copy2 = I_copy2.flatten()
            I_copy2 = I_copy2[I_copy2 > -1]

        return I_unique,I_copy1,I_copy2
    
    
    ########################################## DtN multidomain build and solve ###################################
        
    def get_DtNs(self,device,mode='build',data=0,ff_body_func=None,ff_body_vec=None,uu_true=None):
        """
        Organizes and batches linear algebra operations to be run on GPUs.
        Most prominently this is for the assembly of DtN maps used to make the sparse system,
        but it also handles interior leaf solves and the computation of the RHS in some cases.
        """
        p = self.p; q = self.q; nboxes = self.nboxes; d = self.d
        pdo = self.pdo
        
        # For Gaussian we might need p^2, not (p-2)^2:
        size_face = q**(d-1)
        if (mode == 'build'):
            DtNs = torch.zeros(nboxes,2*d*size_face,2*d*size_face)
            data = torch.zeros(nboxes,2*d*size_face,1)
            #print("Initialized arrays of zeros")
        elif (mode == 'solve'):
            DtNs = torch.zeros(nboxes,p**d,2*data.shape[-1])
        elif (mode == 'reduce_body'):
            DtNs = torch.zeros(nboxes,2*d*size_face,1)
        
        xxloc = self.grid_xx.to(device)
        Nx    = torch.tensor(self.H.Nx).to(device)
        Nxc   = torch.tensor(self.H.Nxc).to(device)
        Jx    = torch.tensor(self.H.JJ.Jx).to(device)
        Jc    = torch.tensor(self.H.JJ.Jc).to(device)
        Jxreo = torch.tensor(self.H.JJ.Jxreorder).to(device)
        Jxun  = torch.tensor(self.H.JJ.Jxunique).to(device)
        Intmap_rev = torch.tensor(self.H.Interp_mat_reverse).to(device)
        Intmap_unq = torch.tensor(self.H.Interp_mat_unique).to(device)
            
        Intmap = torch.tensor(self.H.Interp_mat).to(device)
        Ds     = self.H.Ds.to(device)
        #print("Copied all required parts to device")
        if (mode =='solve'):
            data = data.to(device)

        args = p,q,d,xxloc,Nx,Nxc,Jx,Jc,Jxreo,Jxun,Ds,Intmap,Intmap_rev,Intmap_unq,pdo
        
        # reserve at most 1GB memory for stored DtNs at a time
        f = 0.8e9 # 1 * 0.8 = 0.8 GB in bytes
        if mode == 'solve':
            chunk_max = int(f / ((p**d)*2*data.shape[-1] * 8)) # Size of leaf solution * # RHS * number of bytes per double
        elif mode == 'reduce_body':
            chunk_max = int(f / ((2*d*size_face) * 8)) # Size of reduction * number of bytes per double
        else: #if mode == 'build'
            chunk_max = int(f / ((2*d*size_face)**2 * 8)) # Size of DtN matrix * number of bytes per double
        chunk_size = chunk_max #leaf_ops.get_nearest_div(nboxes,chunk_max)
        
        Aloc_chunkinit = np.min([50,int(nboxes/4)])
        if d==3:
            Aloc_chunkinit = np.max([int(0.2e9 / ((q**6 + 12*q**5 + 72*q**4) * 8)), 1])

        # TODO: replace this with a while loop, end when index reaches nboxes
        j = 0
        while j < nboxes:
            chunk_size = min(chunk_max, nboxes - j)

            DtNs[j:j+chunk_size],Aloc_chunklist = \
            leaf_ops.get_DtNs_helper(*args,j,j+chunk_size, Aloc_chunkinit,device,\
                                    mode,self.interpolate,data,ff_body_func,ff_body_vec,uu_true)

            #print("Did chunk " + str(j))
            Aloc_chunkinit = int(Aloc_chunklist[0])
            j += chunk_size
        return DtNs

    # Input: uu_sol on I_unique
    def solve(self,device,uu_sol,ff_body_func=None,ff_body_vec=None,uu_true=None):
        """
        Given the solution to the subdomain boundaries (present in uu_sol), this computes
        the solution on the subdomain interiors. It also does error analysis if a true solution is known.
        """
        nrhs     = uu_sol.shape[-1]
        size_ext = 4*(self.q)
        if self.d==3:
            size_ext = 6*(self.q**2)

        nboxes   = torch.prod(self.n)
        uu_sol   = uu_sol.to(device)
        
        # Put the solution on all subdomain boundaries (inclduing global DBC) into one array:
        uu_sol_bnd = torch.zeros(nboxes*size_ext,nrhs,device=device)
        uu_sol_bnd[self.I_unique] = uu_sol
        uu_sol_bnd[self.I_copy2]  = uu_sol_bnd[self.I_copy1]
        
        # Compute the subdomain interiors using get_DtNs, then flatten:
        uu_sol_bnd  = uu_sol_bnd.reshape(nboxes,size_ext,nrhs)
        uu_sol_tot  = self.get_DtNs(device,mode='solve',data=uu_sol_bnd,ff_body_func=ff_body_func,ff_body_vec=ff_body_vec,uu_true=uu_true)
        uu_sol_flat = uu_sol_tot[...,:nrhs].flatten(start_dim=0,end_dim=-2)

        resvec_blocks = torch.linalg.norm(uu_sol_tot[...,nrhs:])
        res_lochps = torch.max(resvec_blocks).item()
        return uu_sol_flat, res_lochps
    
    def reduce_body(self,device,ff_body_func,ff_body_vec):
        """
        This forms the RHS using the body load. GPUs are used to compute the statically-condensed load
        on the subdomain boundaries more efficiently.
        """
        ff_red = self.get_DtNs(device,mode='reduce_body',ff_body_func=ff_body_func,ff_body_vec=ff_body_vec)
        
        ff_red_flatten = ff_red.flatten(start_dim=0,end_dim=-2)
        ff_red_flatten[self.I_copy1] += ff_red_flatten[self.I_copy2]
        return ff_red_flatten[self.I_unique]
