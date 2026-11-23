"""Export the new reference, controlled-factorial and sensitivity evidence."""
from pathlib import Path
import json,sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

out=ROOT/'outputs/reviewer_strengthening';figs=out/'figures';figs.mkdir(exist_ok=True)
ref=json.loads((out/'reference_accuracy.json').read_text());summary=json.loads((out/'factorial_summary.json').read_text());rows=json.loads((out/'factorial_metrics.json').read_text());sens=json.loads((out/'retention_sensitivity.json').read_text())['rows']
plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False,'pdf.fonttype':42,'savefig.dpi':300})
def finish(f,n,name):
    f.savefig(figs/f'Fig_{n}_{name}.png',dpi=300,facecolor='white');f.savefig(figs/f'Fig_{n}_{name}.pdf');plt.close(f)
f,ax=plt.subplots(1,2,figsize=(9,3.7),layout='constrained');f.suptitle('Fig. 26. Independent reference and numerical bias',fontweight='bold')
r=ref['spatial_and_temporal_refinement'][:3];ax[0].loglog([v['nx'] for v in r],[100*v['relative_l2_vs_finest'] for v in r],'o-');ax[0].set(xlabel='Longitudinal reference cells',ylabel='Space–time L2 difference (%)',title='Versus 12800 cells, Δt ≤ 0.05 s')
a=np.load(out/'independent_reference_fields.npz');b=np.load(ROOT/'outputs/verification_v5/baseline_fields.npz');truth=a['reference'];p=a['neural_gauss']
for pred,label in [(b['reference'],'Original 800 × 92 FVM'),(p,'Neural cell averages')]:
    err=np.linalg.norm((pred-truth).reshape(61,-1),axis=1)/np.linalg.norm(truth.reshape(61,-1),axis=1);ax[1].plot(a['times'],100*err,label=label)
ax[1].set(xlabel='Time (s)',ylabel='Relative L2 error (%)',title='Against refined reference');ax[1].legend();finish(f,26,'independent_reference')
f,axs=plt.subplots(1,2,figsize=(10,4),layout='constrained');f.suptitle('Fig. 27. Controlled pointwise-PDE / mass-label factorial',fontweight='bold')
combos=[(0,0),(0,20),(4,0),(4,20)];labels=['PDE off\nMass labels off','PDE off\nMass labels on','PDE on\nMass labels off','PDE on\nMass labels on']
for ax,scope in zip(axs,['supported','holdout']):
    means=[]
    for j,(pde,mass) in enumerate(combos):
        seedmeans=[100*np.mean([r['refined_spacetime_l2'] for r in rows if r['scope']==scope and r['seed']==seed and r['pde_weight']==pde and r['mass_snapshot_weight']==mass]) for seed in [2026,2027,2028]]
        means.append(np.mean(seedmeans));ax.scatter(j+np.array([-.08,0,.08]),seedmeans,c='black',s=18,zorder=3)
    ax.bar(range(4),means,color=['#C9D4DC','#A9BDC8','#438EAB','#21678B']);ax.set(xticks=range(4),xticklabels=labels,ylabel='Mean space–time L2 error (%)',title=scope.title()+' configurations');ax.set_ylim(0,6.2)
finish(f,27,'controlled_factorial')
f,axs=plt.subplots(1,2,figsize=(9,3.7),layout='constrained');f.suptitle('Fig. 28. Differentiable final tracer-retention sensitivity',fontweight='bold')
x=[r['discharge_m3_s'] for r in sens]
axs[0].plot(x,[r['reference_retained_fraction'] for r in sens],'s--',label='Refined reference');axs[0].plot(x,[r['neural_retained_fraction'] for r in sens],'o-',label='Frozen hybrid');axs[0].set(xlabel='Discharge (m³/s)',ylabel='Mass remaining / initial mass',title='Plain terrain; t = 300 s');axs[0].legend()
axs[1].plot(x,[r['reference_finite_difference_derivatives']['0.0025']/1200 for r in sens],'s--',label='Reference finite difference');axs[1].plot(x,[r['autograd_dretention_dmultiplier']/1200 for r in sens],'o-',label='Neural automatic derivative');axs[1].set(xlabel='Discharge (m³/s)',ylabel='d(retained fraction)/dQ ((m³/s)⁻¹)',title='Derivative errors: 2.20–3.93%');axs[1].legend();finish(f,28,'retention_sensitivity')

lines=['# Reviewer-strengthening results','Protocol: `protocol.json`. All original checkpoints remain frozen; nine new final checkpoints complete a three-seed 2×2 factorial. No evaluation-based checkpoint selection was performed.','## Independent reference',f"Baseline neural cell-average error is {100*ref['neural_gauss_averages_vs_refined_spacetime']:.4f}% in space–time and {100*ref['neural_gauss_averages_vs_refined_final']:.4f}% at final time against the refined reference. These replace any interpretation of 1.2318% as a PDE-accuracy claim; that value remains agreement with the original upwind reference.",'The separable centered-flux Crank–Nicolson reference is specific to constant coefficients and the prescribed separable initial condition. It is not an exact solution. The finest self-difference zero is not evidence of exactness. Tiny negative values are reported floating-point effects, not clipped outputs.','## Controlled comparison','All factorial cells retain initial-condition, boundary and physical integral-mass penalties. The factors are the pointwise PDE residual and additional numerical mass labels.','| Scope | PDE weight | Mass-label weight | Mean refined L2 (%) | Max refined L2 (%) | Max budget (%) |','|---|---:|---:|---:|---:|---:|']
for r in summary['groups']:lines.append(f"| {r['scope']} | {r['pde_weight']} | {r['mass_snapshot_weight']} | {100*r['mean_refined_l2']:.4f} | {100*r['max_refined_l2']:.4f} | {100*r['max_mass_budget']:.4f} |")
lines+=['The pointwise PDE term improves refined-reference error for every paired seed at both mass-label levels. Additional numerical mass labels worsen refined-reference error in this experiment, even where they improve agreement with the upwind reference. This is evidence of a reference-dependent ranking, not proof that mass supervision is generally harmful. Scenario rows are correlated; the three seeds are the replication units. No significance claim is made.','## Practical sensitivity demonstration','For the frozen full hybrid, retained-mass derivative errors at discharge multipliers 0.85, 1.0 and 1.15 are 3.93%, 3.91% and 2.20%. The demonstration is restricted to the fixed plain-terrain Manning path and initial pulse; it is not a field-control recommendation.','## Interpretation and novelty','The contribution is an experimentally supported verification protocol that distinguishes agreement with a discretized teacher, accuracy against a converged independent method, physical mass budgets, and derivative utility. Hard boundaries, physics losses, mass penalties and numerical-bias concerns are established ideas; no first-of-kind algorithm claim is made.','## Reproduction','Run `python -m scripts.reviewer_ablation --seed SEED` for each of 2026/2027/2028; then `python -m scripts.reviewer_reference_audit`, `python -m scripts.reviewer_evaluate` and `python -m scripts.plot_reviewer_results`. Existing checkpoints/results are reused only where explicitly documented.']
(out/'RESULTS.md').write_text('\n\n'.join(lines),encoding='utf-8')
print('Saved Figures 26–28 and RESULTS.md')
