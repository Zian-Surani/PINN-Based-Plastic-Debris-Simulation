"""Predeclared 2x2 PDE-penalty / mass-label factorial, keeping other losses fixed."""
import argparse,json,hashlib,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from core.training import train_checkpoint,refine_checkpoint

def run(seed):
    out=ROOT/'outputs/reviewer_strengthening/ablation';out.mkdir(parents=True,exist_ok=True)
    # Saved full-hybrid Adam states may be reused: mass labels only enter refinement.
    tasks=[(4,0,ROOT/f'outputs/checkpoints/v5/hybrid_{seed}.pt')]
    source=out/f'pde0_{seed}_adam.pt'
    if not source.exists():train_checkpoint(source,seed=seed,iterations=6000,variant='hybrid',threads=2,pde_weight=0)
    tasks += [(0,0,source),(0,20,source)]
    for pde,mass,source in tasks:
        target=out/f'pde{pde}_mass{mass}_{seed}_final.pt'
        if not target.exists():refine_checkpoint(source,target,max_iter=300,mass_snapshot_weight=mass)
    print('Completed seed',seed,flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--seed',type=int,required=True);run(p.parse_args().seed)
