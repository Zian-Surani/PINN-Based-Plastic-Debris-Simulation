import unittest
import numpy as np
import torch
from scipy.optimize import brentq
from config import make_experiment
from simulation.separable_reference import longitudinal,solve_separable
from core.training import train_checkpoint,refine_checkpoint
from core.pinn_inference import load_pinn
from core.sensitivity import retention_sensitivity

class ReviewerMethods(unittest.TestCase):
    def test_analytical_eigenmode_convergence_and_balance(self):
        e=make_experiment(length=10,width=5,u=.2,diffusion=.5,horizon=2,n_times=2)
        alpha=e.u/(2*e.diffusion)
        k=brentq(lambda k:k*np.cos(k*e.length)+alpha*np.sin(k*e.length),np.pi/(2*e.length),np.pi/e.length)
        initial=lambda x:np.exp(alpha*x)*np.sin(k*x)
        errors=[]
        for nx in [50,100,200]:
            r=longitudinal(e,nx,.002,initial_function=initial)
            # Integrate the exact eigenfunction over cells with matching Gauss rule.
            q,w=np.polynomial.legendre.leggauss(6);dx=e.length/nx
            exact=(initial(r['x'][:,None]+q*dx/2)@(w/2))*np.exp(-(e.diffusion*k*k+e.u**2/(4*e.diffusion))*2)
            errors.append(np.linalg.norm(r['frames'][-1]-exact)/np.linalg.norm(exact))
            self.assertLess(r['relative_balance'],1e-10)
        self.assertLess(errors[-1],errors[0]/10)
        self.assertLess(errors[-1],1e-4)
    def test_reference_rejects_coarse_centered_advection(self):
        with self.assertRaises(ValueError):longitudinal(make_experiment(),200)
    def test_reference_output_and_nonnegative(self):
        e=make_experiment(length=10,width=5,u=.2,diffusion=.5,horizon=2,nx=20,ny=10,n_times=3)
        r=solve_separable(e,nx=200,dt_max=.002)
        self.assertEqual(r['output_frames'].shape,(3,20,10))
        self.assertGreaterEqual(r['minimum'],-1e-12)
    def test_invalid_ablation_weights_fail_before_io(self):
        for bad in [-1,float('nan'),float('inf')]:
            with self.assertRaises(ValueError):train_checkpoint('unused.pt',pde_weight=bad)
            with self.assertRaises(ValueError):refine_checkpoint('absent.pt','unused.pt',mass_snapshot_weight=bad)
    def test_retention_derivative_matches_finite_difference(self):
        model=load_pinn();h=1e-4
        middle=retention_sensitivity(model,1.)
        fd=(retention_sensitivity(model,1+h)['retained_mass_fraction']-retention_sensitivity(model,1-h)['retained_mass_fraction'])/(2*h)
        self.assertAlmostEqual(middle['derivative_per_multiplier'],fd,places=5)
        self.assertEqual(next(model.parameters()).dtype,torch.float32)
    def test_retention_rejects_unaudited_domain(self):
        for v in [.7,1.3,float('nan')]:
            with self.assertRaises(ValueError):retention_sensitivity(None,v)
