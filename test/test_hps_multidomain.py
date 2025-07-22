import numpy as np
import scipy
import torch

from hps.pdo               import PDO_2d,PDO_3d,const
from hps.geom              import BoxGeometry
from hps.domain_driver     import Domain_Driver

import matplotlib.pyplot as plt

def get_discretization_relerr(a,p,kh,ndim,elongated_x=False,elongated_y=False,sparse_assembly='reduced_gpu',solver_type='MUMPS'):

	# Check CUDA availability and adjust settings accordingly
	print("CUDA available %s" % torch.cuda.is_available())
	if (torch.cuda.is_available()):
		print("--num cuda devices %d" % torch.cuda.device_count())
	if ((not torch.cuda.is_available()) and (sparse_assembly == 'reduced_gpu')):
		sparse_assembly = 'reduced_cpu'
		print("Changed sparse assembly to reduced_cpu")

	if (ndim == 2):
		pdo         = PDO_2d(c11=const(1.0),c22=const(1.0),c=const(-kh**2))
		if (elongated_x):
			box         = torch.tensor([[0,0],[2*a,1.0]]) #np.array([[0,0],[2*a,1.0]])
		elif (elongated_y):
			box         = torch.tensor([[0,0],[1.0,2*a]]) #np.array([[0,0],[1.0,2*a]])
		else:
			box         = torch.tensor([[0,0],[1.0,1.0]]) #np.array([[0,0],[1.0,1.0]])
	else:
		pdo         = PDO_3d(c11=const(1.0),c22=const(1.0),c33=const(1.0),c=const(-kh**2))
		box         = torch.tensor([[0,0,0],[0.5,1.0,0.25]]) #np.array([[0,0,0],[0.5,1.0,0.25]])


	geom = BoxGeometry(box)

	solver    = Domain_Driver(geom,pdo,0,a,p,d=ndim)
	solver.build(sparse_assembly, solver_type,verbose=False)
	return  solver.verify_discretization(kh)


def test_hps_2d_elongated():

	a = 1/8; p = 20; kh = 0; ndim = 2
	relerr = get_discretization_relerr(a,p,kh,ndim,elongated_x=True)
	print(f"Relative error for 2D elongated Poisson is {relerr}")
	assert relerr < 1e-12

	kh = 10
	relerr = get_discretization_relerr(a,p,kh,ndim,elongated_y=True)
	print(f"Relative error for 2D elongated Helmholtz with kh={kh} is {relerr}")
	assert relerr < 1e-12


def test_hps_2d():

	a = 1/16; p = 10; kh = 0; ndim = 2
	relerr = get_discretization_relerr(a,p,kh,ndim)
	print(f"Relative error for 2D Poisson is {relerr}")
	assert relerr < 1e-12

	kh = 8
	relerr = get_discretization_relerr(a,p,kh,ndim)
	print(f"Relative error for 2D Helmholtz with kh={kh} is {relerr}")
	assert relerr < 1e-11


def test_hps_3d():

	a = 1/16; p = 7; kh = 0; ndim = 3
	relerr = get_discretization_relerr(a,p,kh,ndim)
	print(f"Relative error for 3D Poisson is {relerr}")
	assert relerr < 1e-12

	kh = 10
	relerr = get_discretization_relerr(a,p,kh,ndim)
	print(f"Relative error for 3D Helmholtz with kh={kh} is {relerr}")
	assert relerr < 5e-8


test_hps_2d()
test_hps_2d_elongated()
test_hps_3d()