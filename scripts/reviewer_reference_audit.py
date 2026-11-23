"""Independent reference convergence and differentiable retention sensitivity."""
from pathlib import Path
import sys,json,hashlib
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from dataclasses import replace
import numpy as np
import torch
from config import make_experiment,initial_normalized
from core.pinn_inference import load_pinn
from simulation.separable_reference import solve_separable,longitudinal

def rel(a,b):return float(np.linalg.norm(a-b)/np.linalg.norm(b))

def run():
    out=ROOT/'outputs/reviewer_strengthening';out.mkdir(exist_ok=True)
    e=make_experiment();base=np.load(ROOT/'outputs/verification_v5/baseline_fields.npz')
    runs={}
    for nx,dt in [(3200,.1),(6400,.1),(12800,.1),(12800,.05)]:
        r=solve_separable(e,nx,dt);runs[(nx,dt)]=r
        print('Reference',nx,dt,'balance',r['relative_balance'],'minimum',r['minimum'],flush=True)
    best=runs[(12800,.05)];truth=best['output_frames']
    rows=[]
    for (nx,dt),r in runs.items():
        rows.append(dict(nx=nx,dt_max=dt,relative_l2_vs_finest=rel(r['output_frames'],truth),final_relative_l2_vs_finest=rel(r['output_frames'][-1],truth[-1]),mass_budget=r['relative_balance'],minimum=r['minimum']))
    report={'method':'Separable constant-coefficient reference; centered conservative x-fluxes and Crank-Nicolson; exact transverse cosine evolution; Gauss cell-average initialization. Not an exact PDE solution.','spatial_and_temporal_refinement':rows,'fvm800_vs_refined_spacetime':rel(base['reference'],truth),'fvm800_vs_refined_final':rel(base['reference'][-1],truth[-1]),'neural_center_samples_vs_refined_cell_averages_spacetime':rel(base['prediction'],truth),'neural_center_samples_vs_refined_cell_averages_final':rel(base['prediction'][-1],truth[-1]),'sampling_caveat':'Legacy metrics compare neural cell-centre samples with FVM cell averages. Additional neural Gauss cell averages are compared below.'}
    torch.set_num_threads(2);model=load_pinn().double()
    # Match cell-average representation using four Gauss nodes per output cell.
    gx,gy=np.meshgrid((np.arange(e.nx)+.5)/e.nx,(np.arange(e.ny)+.5)/e.ny,indexing='ij')
    nod=1/np.sqrt(3);avg=[]
    for time in e.times():
        c=np.zeros((e.nx,e.ny))
        for sx in [-1,1]:
            for sy in [-1,1]:
                zz=np.column_stack([(gx+sx*nod/(2*e.nx)).ravel(),(gy+sy*nod/(2*e.ny)).ravel(),np.full(gx.size,time/e.horizon),np.tile(e.coefficients,(gx.size,1))])
                with torch.no_grad():c+=model(torch.tensor(zz)).numpy().reshape(gx.shape)*e.initial_density/4
        avg.append(c)
    avg=np.asarray(avg);report['neural_gauss_averages_vs_refined_spacetime']=rel(avg,truth);report['neural_gauss_averages_vs_refined_final']=rel(avg[-1],truth[-1])
    np.savez_compressed(out/'independent_reference_fields.npz',reference=truth,neural_gauss=avg,times=e.times(),x=e.grid()[0],y=e.grid()[1])
    (out/'reference_accuracy.json').write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2),flush=True)
    norm=float(best['mass'][0]/e.length)
    xx=torch.tensor(gx.ravel());yy=torch.tensor(gy.ravel());ones=torch.ones_like(xx)
    def neural_retention(mult):
        m=torch.tensor(float(mult),dtype=torch.float64,requires_grad=True)
        a,bx,by=e.coefficients
        z=torch.stack([xx,yy,ones,ones*a*m**.4,ones*bx*m,ones*by*m],dim=1)
        val=model(z).mean()/norm
        grad=torch.autograd.grad(val,m)[0]
        return float(val.detach()),float(grad.detach())
    def numerical_retention(mult):
        exp=replace(e,u=e.u*mult**.4,diffusion=e.diffusion*mult)
        rr=longitudinal(exp,12800,.1,times=[0,e.horizon]);return float(rr['mass'][-1]/rr['mass'][0])
    sens=[]
    for mult in [.85,1.,1.15]:
        nv,grad=neural_retention(mult);rv=numerical_retention(mult)
        ng=(neural_retention(mult+1e-4)[0]-neural_retention(mult-1e-4)[0])/(2e-4)
        fd={}
        for h in [.005,.0025]:fd[str(h)]=(numerical_retention(mult+h)-numerical_retention(mult-h))/(2*h)
        dg=fd['0.0025']
        row={'discharge_multiplier':mult,'discharge_m3_s':1200*mult,'neural_retained_fraction':nv,'reference_retained_fraction':rv,'autograd_dretention_dmultiplier':grad,'neural_finite_difference_derivative':ng,'reference_finite_difference_derivatives':fd,'derivative_relative_error':abs(grad-dg)/abs(dg),'retention_absolute_error':abs(nv-rv),'reference_step_relative_change':abs(fd['0.005']-dg)/abs(dg)}
        sens.append(row);print('Sensitivity',row,flush=True)
    (out/'retention_sensitivity.json').write_text(json.dumps({'scope':'Frozen seed-2026 hybrid; final retained-mass fraction at 300s, plain terrain, discharge along Manning relationship; no retraining or runtime claim. Derivatives are with respect to Q/Qbase; divide by 1200 for per-(m3/s) derivatives. Neural diagnostic uses float64 evaluation of frozen float32-trained weights.','rows':sens},indent=2))

if __name__=='__main__':run()
