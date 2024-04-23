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
    
    def __init__(self, pdo, domain, a, p, d):
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
        
        n = (self.domain[:,1] - self.domain[:,0]) / (2*self.a)
        n = torch.round(n).int()
        nboxes = torch.prod(n)
        self.n = n; self.nboxes = nboxes
        self.H = hps_disc.HPS_Disc(a,p,d)
        self.hmin = self.H.hmin
        
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
        
        grid_xx = self.get_grid()
        self.grid_xx = grid_xx
        
        Jx = torch.tensor(self.H.JJ.Jx)
        Jc = torch.tensor(self.H.JJ.Jc)

        self.grid_ext = self.grid_xx[:,Jx,:].flatten(start_dim=0,end_dim=-2)
        self.xx_ext = self.grid_ext.flatten(start_dim=0,end_dim=-2)
        #print(Jx)
        #print(self.grid_ext)
        #print(self.grid_ext.shape)
        
        if d==2:
            self.I_unique, self.I_copy1, self.I_copy2 = self.get_unique_inds()
            self.xx_active = self.xx_ext[self.I_unique,:]
            """print(self.I_unique)
            print(self.I_copy1)
            print(self.I_copy2)
            print(self.I_unique.shape)
            print(self.I_copy1.shape)
            print(self.I_copy2.shape)"""
        
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
        tic = time()
        DtN_loc = self.get_DtNs(device,'build')
        toc_DtN = time() - tic;
        
        size_face = self.p-2; size_ext = 4*size_face
        n0,n1 = self.n
        
        tic = time()
        Iext_box  = torch.arange(size_face).repeat(n0*n1,4).reshape(n0*n1,size_ext)
        toc_alloc = time() - tic;
        
        tic = time()
        # Go through range of n
        for Npan_ind in range(n0):
            # This increases in each level
            npan_offset = (2*n1+1)*Npan_ind
            
            for box_j in range(n1):
                
                box_ind = Npan_ind*n1 + box_j
                
                l_ind = npan_offset + box_j
                r_ind = npan_offset + 2*n1 + 1 + box_j
                d_ind = npan_offset + n1 + box_j
                u_ind = npan_offset + n1 + box_j + 1
                Iext_box[box_ind,:size_face]              += l_ind * size_face;
                Iext_box[box_ind,size_face:2*size_face]   += r_ind * size_face;
                Iext_box[box_ind,2*size_face:3*size_face] += d_ind * size_face;
                Iext_box[box_ind,3*size_face:]            += u_ind * size_face
        toc_index = time() - tic
        
        tic = time()
        row_data,col_data = batched_meshgrid(n0*n1,size_ext,Iext_box,Iext_box)
        
        # LOOK HERE:
        data = DtN_loc.flatten()
        row_data = row_data.flatten()
        col_data = col_data.flatten()
        toc_flatten = time() - tic
        
        tic = time()
        sp_mat = sp.coo_matrix(( np.array(data),\
                                (np.array(row_data,dtype=int),np.array(col_data,dtype=int)))).tocsr()
        sp_mat = sp_mat.tocsr()
        toc_csr_scipy = time() - tic;
        
        toc_forloop = toc_index + toc_flatten + toc_alloc
        
        if (verbose):
            print("\t--time to do for loop (alloc,index, flatten) (%5.2f,%5.2f,%5.2f)"\
                  %(toc_alloc,toc_index,toc_flatten))
            print("\t--time to assemble sparse HPS (DtN ops, for loop, csr_scipy) (%5.2f,%5.2f,%5.2f)"\
                  %(toc_DtN,toc_forloop,toc_csr_scipy))
        t_dict = dict()
        t_dict['toc_DtN'] = toc_DtN
        t_dict['toc_forloop'] = toc_forloop
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
        xx = torch.zeros(self.nboxes,self.p**self.d, self.d)
        for i in range(n[0]):
            for j in range(n[1]):
                if self.d==2:
                    box   = i*n[1] + j
                    zzloc = zz.clone()
                    zzloc[:,0] += self.a + 2*self.a*i + self.domain[0,0]
                    zzloc[:,1] += self.a + 2*self.a*j + self.domain[1,0]
                    xx[box,:,:] = zzloc
                else:
                    for k in range(n[2]):
                        box   = i*n[1]*n[2] + j*n[2] + k
                        zzloc = zz.clone()
                        zzloc[:,0] += self.a + 2*self.a*i + self.domain[0,0]
                        zzloc[:,1] += self.a + 2*self.a*j + self.domain[1,0]
                        zzloc[:,2] += self.a + 2*self.a*k + self.domain[2,0]
                        xx[box,:,:] = zzloc

        return xx
    
    def get_unique_inds(self):
        """
        Identifies unique and duplicated indices for handling boundary conditions and overlaps between subdomains.
        
        Returns:
        - I_unique, I_copy1, I_copy2 (torch.Tensor): Tensors representing unique and duplicated grid indices.
        """
        if self.d==2:
            size_face = self.p-2
            n0,n1 = self.n

            box_ind = torch.arange(n0*n1*4*size_face).reshape(n0,n1,4*size_face)

            inds_unique = size_face * (n0 * (2*n1+1) + n1)
            inds_rep    = size_face * (n0 * (n1-1) + n1 * (n0-1))

            I_unique   = torch.zeros(inds_unique).long()
            I_copy1    = torch.zeros(inds_rep).long()
            I_copy2    = torch.zeros(inds_rep).long()

            offset_unique = 0; offset_copy = 0
            L_inds_slab0 = box_ind[0,:,:size_face].flatten()
            I_unique[offset_unique : offset_unique + n1 * size_face] = L_inds_slab0
            offset_unique += n1 * size_face

            for i in range(n0):
                
                if (i > 0):
                    # shared L,R face between current panel (L) and previous panel (R) 
                    l_faces_rep = box_ind[i,:,:size_face].flatten()
                    r_faces_rep = box_ind[i-1,:,size_face:2*size_face].flatten()
                    # can use a check here with self.xx to verify these indices line up

                    I_unique[offset_unique : offset_unique + (n1)*size_face] = l_faces_rep
                    offset_unique += (n1) * size_face

                    I_copy1[offset_copy : offset_copy + (n1)*size_face] = l_faces_rep
                    I_copy2[offset_copy : offset_copy + (n1)*size_face] = r_faces_rep
                    offset_copy += (n1) * size_face

                # unique down face
                d_face_uni = box_ind[i,0,2*size_face:3*size_face]
                I_unique[offset_unique : offset_unique + size_face] = d_face_uni
                offset_unique += size_face

                # repeated up and down faces
                u_faces_rep = box_ind[i,:n1-1,3*size_face:].flatten()
                d_faces_rep = box_ind[i,1:,2*size_face:3*size_face].flatten()

                I_unique[offset_unique : offset_unique + (n1-1)*size_face] = d_faces_rep
                offset_unique += (n1-1) * size_face

                I_copy1[offset_copy : offset_copy + (n1-1)*size_face] = d_faces_rep
                I_copy2[offset_copy : offset_copy + (n1-1)*size_face] = u_faces_rep
                offset_copy += (n1-1) * size_face

                # unique up face
                u_face_uni = box_ind[i,n1-1,3*size_face:]
                I_unique[offset_unique : offset_unique + size_face] = u_face_uni
                offset_unique += size_face

            R_inds_slablast = box_ind[n0-1,:,size_face:2*size_face].flatten()
            I_unique[offset_unique : offset_unique + n1 * size_face] = R_inds_slablast
            offset_unique += n1 * size_face
        else:
            size_face = (self.p-2)**2
            # Unique: bottom boundary of bottom layer of boxes, 
        return I_unique,I_copy1,I_copy2
    
    
    ########################################## DtN multidomain build and solve ###################################
        
    def get_DtNs(self,device,mode='build',data=0,ff_body_func=None):
        a = self.a; p = self.p; nboxes = self.nboxes; d = self.d
        pdo = self.pdo
        
        size_face = (p-2)**(d-1)

        if (mode == 'build'):
            DtNs = torch.zeros(nboxes,2*d*size_face,2*d*size_face)
            data = torch.zeros(nboxes,2*d*size_face,1)
        elif (mode == 'solve'):
            DtNs = torch.zeros(nboxes,p**d,2*data.shape[-1])
        elif (mode == 'reduce_body'):
            DtNs = torch.zeros(nboxes,2*d*size_face,1)
        
        xxloc = self.grid_xx.to(device)
        Nxtot = torch.tensor(self.H.Nx).to(device)
        Jx    = torch.tensor(self.H.JJ.Jx).to(device)
        Jc    = torch.tensor(self.H.JJ.Jc).to(device)
        Jxreo = torch.tensor(self.H.JJ.Jxreorder).to(device)
        Intmap= torch.tensor(self.H.Interp_mat).to(device)
        Ds    = self.H.Ds.to(device)
        if (mode =='solve'):
            data = data.to(device)
            
        args = p,d,xxloc,Nxtot,Jx,Jc,Jxreo,Ds,Intmap,pdo
        
        # reserve at most 1GB memory for stored DtNs at a time
        f = 0.8e9 # 0.8 GB in bytes
        chunk_max = int(f / ((2*d*size_face)**2 * 8))
        chunk_size = leaf_ops.get_nearest_div(nboxes,chunk_max)
        
        assert np.mod(nboxes,chunk_size) == 0
        Aloc_chunkinit = np.min([50,int(nboxes/4)])
        for j in range(int(nboxes / chunk_size)):
            #print("Indices: " + str(j*chunk_size) + " to " + str((j+1)*chunk_size))
            DtNs[j*chunk_size:(j+1)*chunk_size],Aloc_chunklist = \
            leaf_ops.get_DtNs_helper(*args,j*chunk_size,(j+1)*chunk_size, Aloc_chunkinit,device,\
                                    mode,data,ff_body_func)
            Aloc_chunkinit = int(Aloc_chunklist[-2])
        #print("DtNs interior = " + str(DtNs[:,Jc]))
        #print("DtNs exterior = " + str(DtNs[:,Jx]))
        #print("Whole DtN = " + str(DtNs))
        return DtNs

    # Input: uu_sol on I_unique
    def solve(self,device,uu_sol,ff_body_func=None):
        nrhs     = uu_sol.shape[-1] # almost always 1, guessing this if for solving multiple rhs in parallel

        size_ext = 4*(self.p-2)
        if self.d==3:
            size_ext = 6*(self.p-2)**2

        nboxes   = torch.prod(self.n)
        uu_sol   = uu_sol.to(device)
        
        uu_sol_bnd = torch.zeros(nboxes*size_ext,nrhs,device=device)
        if self.d==2:
            uu_sol_bnd[self.I_unique] = uu_sol
            uu_sol_bnd[self.I_copy2]  = uu_sol_bnd[self.I_copy1]
        else: # For 3D we don't identify unique and copy bdries of boxes yet, we just set all the box boundaries to true solution:
            uu_sol_bnd[:] = uu_sol
        
        uu_sol_bnd = uu_sol_bnd.reshape(nboxes,size_ext,nrhs)
        #print(uu_sol_bnd)
        uu_sol_tot = self.get_DtNs(device,mode='solve',data=uu_sol_bnd,ff_body_func=ff_body_func)
        #print(uu_sol_tot)
        
        uu_sol_flat = uu_sol_tot[...,:nrhs].flatten(start_dim=0,end_dim=-2)
        if self.d==3:
            print("Flattened diff is " + str(torch.linalg.norm(uu_sol_flat[self.p**3:2*self.p**3] - uu_sol_tot[1,:,:nrhs])))
        resvec_blocks = torch.linalg.norm(uu_sol_tot[...,nrhs:])
        res_lochps = torch.max(resvec_blocks).item()
        return uu_sol_flat, res_lochps
    
    def reduce_body(self,device,ff_body_func):
        ff_red = self.get_DtNs(device,mode='reduce_body',ff_body_func=ff_body_func)
        
        ff_red_flatten = ff_red.flatten(start_dim=0,end_dim=-2)
        ff_red_flatten[self.I_copy1] += ff_red_flatten[self.I_copy2]
        return ff_red_flatten[self.I_unique]