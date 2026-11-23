"""Evaluate all factorial cells without tuning or selecting their checkpoints."""
from pathlib import Path
import sys,json,csv
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np
import torch
from config import all_experiments,initial_field
from scripts.extended_verification import unseen_experiments
from simulation.separable_reference import solve_separable
from simulation.finite_volume import solve
from simulation.validation_suite import compute_error_timeseries,boundary_flux,compute_mass_metrics
from core.pinn_inference import load_pinn,predict_frames

def run():
    out=ROOT/'outputs/reviewer_strengthening';cache=out/'reference_cache';cache.mkdir(exist_ok=True)
    torch.set_num_threads(2);exps=all_experiments()+unseen_experiments();refs={};refrows=[]
    for i,e in enumerate(exps):
        path=cache/f'{i:02d}.npz'
        if path.exists():
            z=np.load(path);fine=z['refined'];old=z['upwind'];row=json.loads(str(z['metadata']))
        else:
            small=solve_separable(e,6400,.1);large=solve_separable(e,12800,.1)
            fine=large['output_frames'];ratio=4;r=solve(e.refined(e.nx*ratio,e.ny*ratio));old=r['frames'].reshape(e.n_times,e.nx,ratio,e.ny,ratio).mean((2,4))
            delta=np.linalg.norm(small['output_frames']-fine)/np.linalg.norm(fine)
            row={'case':i,'terrain':e.terrain,'season':e.season,'spacetime_6400_vs_12800':float(delta),'final_6400_vs_12800':float(np.linalg.norm(small['output_frames'][-1]-fine[-1])/np.linalg.norm(fine[-1])),'centered_mass_budget':large['relative_balance'],'centered_minimum':large['minimum'],'upwind_reference_spacetime_error':float(np.linalg.norm(old-fine)/np.linalg.norm(fine))}
            np.savez_compressed(path,refined=fine,upwind=old,metadata=json.dumps(row))
        refs[i]=(fine,old);refrows.append(row);print('Reference case',i,row,flush=True)
    (out/'all_reference_convergence.json').write_text(json.dumps(refrows,indent=2))
    rows=[]
    for seed in [2026,2027,2028]:
        for pde,mass in [(0,0),(0,20),(4,0),(4,20)]:
            path=ROOT/f'outputs/checkpoints/v5/hybrid_{seed}_final.pt' if (pde,mass)==(4,20) else out/f'ablation/pde{pde}_mass{mass}_{seed}_final.pt'
            model=load_pinn(path)
            for i,e in enumerate(exps):
                research=i>=12;pred=predict_frames(model,e,research=research);truth,old=refs[i]
                err=compute_error_timeseries(pred,truth);old_err=compute_error_timeseries(pred,old)
                flux,_,_=boundary_flux(model,e,e.times(),research=research);budget=compute_mass_metrics(pred,e,e.times(),flux)
                ic=initial_field(e)
                rows.append({'seed':seed,'pde_weight':pde,'mass_snapshot_weight':mass,'scope':'holdout' if research else 'supported','terrain':e.terrain,'season':e.season,'refined_spacetime_l2':err['relative_l2_spacetime'],'refined_final_l2':err['relative_l2_final'],'upwind_spacetime_l2':old_err['relative_l2_spacetime'],'mass_budget':budget['max_relative_balance_error'],'initial_l2':float(np.linalg.norm(pred[0]-ic)/np.linalg.norm(ic)),'minimum':float(pred.min())})
            print('Evaluated',seed,pde,mass,flush=True)
            (out/'factorial_metrics.json').write_text(json.dumps(rows,indent=2))
    summary=[]
    for scope in ['supported','holdout']:
        for pde,mass in [(0,0),(0,20),(4,0),(4,20)]:
            r=[v for v in rows if v['scope']==scope and v['pde_weight']==pde and v['mass_snapshot_weight']==mass]
            summary.append({'scope':scope,'pde_weight':pde,'mass_snapshot_weight':mass,'cases':len(r),'mean_refined_l2':float(np.mean([v['refined_spacetime_l2'] for v in r])),'max_refined_l2':max(v['refined_spacetime_l2'] for v in r),'mean_upwind_l2':float(np.mean([v['upwind_spacetime_l2'] for v in r])),'max_mass_budget':max(v['mass_budget'] for v in r)})
    # Seed is the replication unit; scenario rows are correlated, not 36 independent seeds.
    paired=[]
    for scope in ['supported','holdout']:
        for seed in [2026,2027,2028]:
            means={(p,m):np.mean([r['refined_spacetime_l2'] for r in rows if r['scope']==scope and r['seed']==seed and r['pde_weight']==p and r['mass_snapshot_weight']==m]) for p,m in [(0,0),(0,20),(4,0),(4,20)]}
            paired.append({'scope':scope,'seed':seed,'pde_effect_no_mass':float(means[(4,0)]-means[(0,0)]),'pde_effect_with_mass':float(means[(4,20)]-means[(0,20)]),'mass_effect_no_pde':float(means[(0,20)]-means[(0,0)]),'mass_effect_with_pde':float(means[(4,20)]-means[(4,0)]),'interaction':float(means[(4,20)]-means[(4,0)]-means[(0,20)]+means[(0,0)])})
    (out/'factorial_summary.json').write_text(json.dumps({'groups':summary,'paired_seed_effects':paired,'interpretation':'Effects are signed changes in mean refined-reference L2; negative is improvement. Three seeds only. All four cells retain IC, boundary and physical integral-mass penalties. This isolates the pointwise PDE term and numerical mass labels, not physics versus no physics. Center samples versus cell-average references as in original protocol; baseline Gauss-average sensitivity is separately audited.'},indent=2))
    with (out/'factorial_metrics.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    print(json.dumps(summary,indent=2),flush=True)

if __name__=='__main__':run()
