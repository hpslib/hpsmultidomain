from pdo import *
import torch
import numpy as np

alpha = 1.0
rotation = 1.0 / np.sqrt(2)

def z1(zz):
    return alpha*zz[:,0]
def z2(zz):
    return alpha*zz[:,1]
def z3(zz):
    return alpha*zz[:,2]

def y1(zz):
    return alpha*zz[:,0]
def y2(zz):
    return alpha*zz[:,1]
def y3(zz):
    return alpha*zz[:,2]

def y1_d1(zz):
    return alpha*rotation*torch.ones(zz.shape[0],device=zz.device)

def y2_d2(zz):
    return alpha*rotation*torch.ones(zz.shape[0],device=zz.device)

def y3_d3(zz):
    return alpha*rotation*torch.ones(zz.shape[0],device=zz.device)

def bfield_constant(xx,kh):
    return -(kh**2) * torch.ones(xx.shape[0],device=xx.device)

# Optional: to test rotation
def y1_d2(zz):
    return -alpha*rotation*torch.ones(zz.shape[0],device=zz.device)

def y2_d1(zz):
    return alpha*rotation*torch.ones(zz.shape[0],device=zz.device)


pdo, parameter_map, inv_parameter_map = pdo_param_3d(0, bfield_constant, z1, z2, z3, y1, y2, y3, y1_d1=y1_d1, y2_d2=y2_d2, y3_d3=y3_d3, y1_d2=y1_d2, y2_d1=y2_d1)

# Test on data:
xx = torch.tensor([[0, 1, 0, 1, 0, 1, 0, 1],[0, 0, 1, 1, 0, 0, 1, 1], [0, 0, 0, 0, 1, 1, 1, 1]])
xx = torch.transpose(xx, 0, 1)

print("c1, c2, c3:")
print(pdo.c1(xx))
print(pdo.c2(xx))
print(pdo.c3(xx))

print("c11, c22, c33:")
print(pdo.c11(xx))
print(pdo.c22(xx))
print(pdo.c33(xx))

print("c12, c13, c23:")
print(pdo.c12(xx))
print(pdo.c13(xx))
print(pdo.c23(xx))

print("c:")
print(pdo.c(xx))