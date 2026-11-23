"""Differentiable final-mass retention for the explicitly audited hydraulic path."""
from copy import deepcopy
import math
import numpy as np
import torch
from scipy.integrate import quad
from config import make_experiment
from core.pinn_inference import validate_experiment

def retention_sensitivity(model,discharge_multiplier):
    mval=float(discharge_multiplier)
    if not math.isfinite(mval) or not .85<=mval<=1.15:
        raise ValueError('Sensitivity demonstration is restricted to plain terrain and Q/Qbase in [0.85,1.15]')
    exp=make_experiment();validate_experiment(model,exp)
    net=deepcopy(model).double().eval()
    x,y=np.meshgrid((np.arange(exp.nx)+.5)/exp.nx,(np.arange(exp.ny)+.5)/exp.ny,indexing='ij')
    xx=torch.tensor(x.ravel());yy=torch.tensor(y.ravel());one=torch.ones_like(xx)
    mult=torch.tensor(mval,dtype=torch.float64,requires_grad=True)
    a,bx,by=exp.coefficients
    features=torch.stack([xx,yy,one,one*a*mult**.4,one*bx*mult,one*by*mult],dim=1)
    def pulse(xi):
        r=(xi-.25)/.15
        return math.exp(1-1/(1-r*r)) if abs(r)<1 else 0.
    # Normalize a reported integral quantity, never rescale the predicted field.
    initial_mean=quad(pulse,.1,.4,epsabs=1e-13,epsrel=1e-13)[0]
    retention=net(features).mean()/initial_mean
    gradient=torch.autograd.grad(retention,mult)[0]
    return {'discharge_m3_s':1200*mval,'discharge_multiplier':mval,'retained_mass_fraction':float(retention.detach()),'derivative_per_multiplier':float(gradient.detach()),'derivative_per_m3_s':float(gradient.detach())/1200,'horizon_s':300,'seed':model.metadata['seed'],'scope':'Synthetic plain-terrain Manning path, fixed pulse, 300 s. Float64 evaluation of frozen weights; spatial midpoint mass quadrature. Not a field forecast or an exact PDE derivative.'}
