"""Generate manuscript illustrations from frozen outputs; no scientific code edits."""
from pathlib import Path
import sys,json,hashlib,csv,zipfile,argparse
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle,FancyBboxPatch
from matplotlib.backends.backend_pdf import PdfPages
from config import make_experiment,initial_field,TERRAIN,SEASON_DISCHARGE
from core.pinn_inference import load_pinn,predict_frames
from simulation.validation_suite import evaluate,boundary_flux,compute_mass_metrics,compute_error_timeseries
from simulation.finite_volume import solve
from simulation.uncertainty_quantification import residual_interval_audit

parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,default=ROOT/'docx'/'replacement_figures');args=parser.parse_args()
OUT=args.output;OUT.mkdir(parents=True,exist_ok=True)
DATA=OUT/'data';DATA.mkdir(exist_ok=True)
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.titlesize':11,'axes.labelsize':10,'legend.fontsize':9,'figure.dpi':120,'savefig.dpi':300,'axes.spines.top':False,'axes.spines.right':False,'pdf.fonttype':42,'svg.fonttype':'none'})
BLUE='#176B96';ORANGE='#D46B28';GREEN='#38856A'
def read(p):return json.loads((ROOT/p).read_text(encoding='utf-8'))
torch.set_num_threads(2)
e=make_experiment();model=load_pinn();z=np.load(ROOT/'outputs/verification_v5/baseline_fields.npz')
pred,ref,t,x,y=[z[k] for k in ['prediction','reference','times','x','y']]
errors=compute_error_timeseries(pred,ref);metrics=read('outputs/verification_v5/metrics.json')
saved=next(r for r in metrics if r['variant']=='hybrid' and r['seed']==2026 and r['terrain']=='plain' and r['season']=='normal')
assert np.isclose(errors['relative_l2_spacetime'],saved['relative_l2_spacetime'],rtol=1e-10)
flux,_,_=boundary_flux(model,e,t);mass=compute_mass_metrics(pred,e,t,flux)
assert np.isclose(mass['max_relative_balance_error'],saved['mass_balance_relative_error'],rtol=1e-6)
fine=solve(e.refined(800,92));assert np.allclose(fine['frames'].reshape(61,200,4,23,4).mean((2,4)),ref,rtol=1e-10,atol=1e-15)
me=make_experiment('mountain','wet');mount=evaluate(model,me)
ms=next(r for r in metrics if r['variant']=='hybrid' and r['seed']==2026 and r['terrain']=='mountain' and r['season']=='wet')
assert np.isclose(mount['metrics']['relative_l2_spacetime'],ms['relative_l2_spacetime'],rtol=1e-6)
data_pred=predict_frames(load_pinn(ROOT/'outputs/checkpoints/v5/data_only_2026_final.pt'),e)
np.savez_compressed(DATA/'plotted_fields.npz',times=t,x=x,y=y,prediction=pred,reference=ref,data_only=data_pred,mountain_prediction=mount['prediction'],mountain_reference=mount['reference'],mountain_x=me.grid()[0],mountain_y=me.grid()[1],mass=mass['mass_kg'],export=mass['integrated_outward_flux_kg'],budget=mass['balance_error_kg'],reference_mass=fine['mass_kg'],reference_export=fine['escaped_mass_kg'],reference_budget=fine['balance_error_kg'])
timing=read('outputs/verification_v5/timings.json')['hybrid_2026_plain_normal'];uq=residual_interval_audit(pred,ref)
records=[];book=PdfPages(OUT/'ALL_25_FIGURES.pdf')
def canvas(n,title,rows=1,cols=1,size=None):
 f,axs=plt.subplots(rows,cols,figsize=size or (8.2,4.2),squeeze=False,layout='constrained');f.suptitle(f'Fig. {n}. {title}',fontsize=13,fontweight='bold');return f,axs.ravel()
def save(n,f,slug,caption,source):
 name=f'Fig_{n:02d}_{slug}';f.savefig(OUT/(name+'.png'),facecolor='white');f.savefig(OUT/(name+'.pdf'),facecolor='white');book.savefig(f);plt.close(f)
 records.append(dict(figure=n,file=name+'.png',caption=caption,source=source));print('Created',name,flush=True)
def heat(ax,field,exp=e,title='',symmetric=False,vmax=None,colorbar=True):
 mx=float(np.max(np.abs(field))) if vmax is None else vmax
 im=ax.pcolormesh(np.linspace(0,exp.length,exp.nx+1),np.linspace(0,exp.width,exp.ny+1),field.T,shading='flat',cmap='RdBu_r' if symmetric else 'viridis',vmin=-mx if symmetric else 0,vmax=mx)
 ax.set(xlabel='x (m)',ylabel='y (m)',title=title,xlim=(0,exp.length),ylim=(0,exp.width))
 if colorbar:ax.figure.colorbar(im,ax=ax,label='Tracer density error (kg/m²)' if symmetric else 'Tracer density (kg/m²)',format='%.1e')
 return im
def box(ax,xy,w,h,text,color='#EAF2F7'):
 ax.add_patch(FancyBboxPatch(xy,w,h,boxstyle='round,pad=0.015',facecolor=color,edgecolor=BLUE,linewidth=1));ax.text(xy[0]+w/2,xy[1]+h/2,text,ha='center',va='center',fontsize=9)
def arrow(ax,a,b):ax.annotate('',xy=b,xytext=a,arrowprops=dict(arrowstyle='->',lw=1.5,color=BLUE))
BASE='Synthetic plain/normal configuration; seed 2026; 200 × 23 output cells; T=300 s.'
SRC='outputs/verification_v5/baseline_fields.npz'

f,(ax,)=canvas(1,'Transport domain and boundary conditions',size=(8.2,4))
ax.set(xlim=(-180,1180),ylim=(-60,190));ax.add_patch(Rectangle((0,0),1000,118.8,fc='#EEF5F8',ec=BLUE,lw=2));ax.add_patch(Rectangle((100,0),300,118.8,fc='#CBDDB0',alpha=.6));arrow(ax,(450,59.4),(780,59.4));ax.text(600,78,'u = 2.602994 m/s',ha='center');ax.text(-20,60,'Inlet\nC = 0',ha='right',va='center');ax.text(1020,60,'Outlet\n∂C/∂x = 0\nAdvective export',va='center');ax.text(500,137,'Reflecting walls: ∂C/∂y = 0',ha='center');ax.text(250,35,'Initial pulse\n100 < x < 400 m',ha='center');ax.set(xlabel='x (m)',ylabel='y (m)');ax.set_xticks([0,100,400,1000]);ax.set_yticks([0,118.8]);save(1,f,'domain_boundaries','Synthetic transport domain with a clean-water inlet, zero-gradient outlet and reflecting lateral walls. Shading identifies initial-pulse support, not uniform density. Both upstream diffusive export and downstream advective export enter the mass budget.','config.py; simulation/finite_volume.py')
f,(ax,)=canvas(2,'Physical output grid')
for v in np.linspace(0,1000,201):ax.axvline(v,color=BLUE,alpha=.28,lw=.35)
for v in np.linspace(0,118.8,24):ax.axhline(v,color=BLUE,alpha=.45,lw=.5)
ax.set(xlim=(0,1000),ylim=(0,118.8),xlabel='x (m)',ylabel='y (m)');ax.text(500,60,'200 × 23 cell-centred output grid\nΔx = 5 m; Δy = 5.165217 m',ha='center',va='center',bbox=dict(fc='white',alpha=.94,ec=BLUE));save(2,f,'physical_grid','Output-grid cell faces in physical coordinates. The plotted grid is 200 × 23; training labels use 400 × 48 and the accuracy reference uses 800 × 92 cells.','config.py')
f,(ax,)=canvas(3,'Training and verification workflow',size=(9,5.1));ax.axis('off');ax.set(xlim=(-.03,1.03),ylim=(0,1))
box(ax,(.02,.73),.27,.18,'Synthetic physics\nIC, BC, terrain and season');box(ax,(.37,.73),.27,.18,'400 × 48 FVM labels\n12 supported configurations');box(ax,(.72,.73),.26,.18,'Hybrid training\nAdam + L-BFGS');arrow(ax,(.29,.82),(.37,.82));arrow(ax,(.64,.82),(.72,.82));box(ax,(.72,.39),.26,.19,'Frozen checkpoints\nSeeds 2026, 2027, 2028');arrow(ax,(.85,.73),(.85,.58));box(ax,(.02,.39),.27,.19,'800 × 92 FVM reference\nAnalytical/refinement audit');arrow(ax,(.15,.73),(.15,.58));box(ax,(.37,.39),.27,.19,'200 × 23 comparison\nField errors + flux budget');arrow(ax,(.29,.48),(.37,.48));arrow(ax,(.72,.48),(.64,.48));box(ax,(.13,.05),.74,.19,'Supported cases + frozen hydraulic interpolation\nMeasured runtime and descriptive residual-interval coverage');arrow(ax,(.5,.39),(.5,.24));save(3,f,'workflow','Implemented training and verification workflow. Numerical labels and the finer evaluation reference have different resolutions. Frozen hydraulic-interpolation tests are separate from supported training configurations; no field-validation dataset is used.','core/training.py; scripts/run_validation.py; scripts/extended_verification.py')
f,(ax,)=canvas(4,'Parameter-conditioned neural architecture',size=(9,3.8));ax.axis('off');ax.set(xlim=(-.03,1.03),ylim=(0,1))
labels=['6 physical inputs\nξ, η, τ, a, bx, by','7 transformed\nfeatures','3 hidden layers\n64 Tanh units each','Softplus ×\ninlet factor','C = 0.001 q\nkg/m²']
for k,label in enumerate(labels):box(ax,(.01+k*.2,.35),.17,.3,label);
for k in range(4):arrow(ax,(.18+k*.2,.5),(.21+k*.2,.5))
ax.text(.5,.14,'Shared positivity and boundary transforms in both hybrid and data-only models',ha='center',fontsize=10);save(4,f,'neural_architecture','Six dimensionless physical inputs are mapped to seven features and passed through three 64-unit Tanh hidden layers. A Softplus output and analytical coordinate/inlet transforms enforce nonnegativity and boundary conditions; initial, PDE and mass constraints remain soft.','core/pinn_model.py')
hist=read('outputs/checkpoints/v5/hybrid_2026.history.json');meta=read('outputs/checkpoints/v5/hybrid_2026_final.json');steps=[v['step'] for v in hist]
f,axs=canvas(5,'Recorded optimization histories',cols=2,size=(9,3.8));axs[0].semilogy(steps,[v['total'] for v in hist],color=BLUE);axs[0].set(xlabel='Adam iteration',ylabel='Recorded total objective',title='Adam: sampled batches, ramped weights');lh=meta['refinement']['loss_history'];axs[1].semilogy(np.arange(1,len(lh)+1),lh,color=ORANGE);axs[1].set(xlabel='L-BFGS closure evaluation',ylabel='Recorded total objective',title='L-BFGS: fixed samples, line search');save(5,f,'optimization_histories','Recorded hybrid optimization objectives for seed 2026. Adam and L-BFGS use different sampling/objective details and are plotted separately. Closure evaluations include line-search trials and are not optimizer iterations; no monotone convergence is implied.','outputs/checkpoints/v5/hybrid_2026.history.json; hybrid_2026_final.json')
f,(ax,)=canvas(6,'Conservative finite-volume face fluxes',size=(7.5,4.6));ax.set_aspect('equal')
for i in range(3):
 for j in range(3):ax.add_patch(Rectangle((i,j),1,1,ec='#9DA9AF',fc='#E8F1F6' if (i,j)==(1,1) else 'white'))
ax.text(1.5,1.5,'Cᵢⱼ',ha='center',va='center',fontsize=16)
for a,b,label,xy in [((.7,1.5),(1.2,1.5),'Jxᵢ₋½,ⱼ',(.55,1.85)),((1.8,1.5),(2.3,1.5),'Jxᵢ₊½,ⱼ',(2.45,1.85)),((1.5,.7),(1.5,1.2),'Jyᵢ,ⱼ₋½',(2,.65)),((1.5,1.8),(1.5,2.3),'Jyᵢ,ⱼ₊½',(2,2.4))]:arrow(ax,a,b);ax.text(*xy,label,ha='center')
ax.set(xlim=(-.2,3.2),ylim=(-.2,3.2),xlabel='Schematic cell coordinate (not metres)',ylabel='Schematic cell coordinate (not metres)');ax.set_xticks([]);ax.set_yticks([]);save(6,f,'finite_volume_fluxes','Finite-volume control cell and oriented face fluxes. Interior flux contributions cancel between adjacent cells. Advection uses an upwind face value and diffusion uses a central face gradient. Diagram is schematic, not a physical grid.','simulation/finite_volume.py')
ana=read('outputs/extended_verification_v5/analytical.json')['spatial'];f,(ax,)=canvas(7,'Independent analytical convergence');ns=np.array([r['nx'] for r in ana]);vals=np.array([r['relative_l2'] for r in ana]);ax.loglog(ns,vals*100,'o-',color=BLUE,label='Measured final relative L2');ax.loglog(ns,vals[0]*100*ns[0]/ns,'--',color=ORANGE,label='First-order guide');ax.set(xlabel='Longitudinal cells, nx (ny = nx/2)',ylabel='Final relative L2 error (%)');ax.set_xticks(ns,labels=ns);ax.legend();ax.grid(alpha=.2);save(7,f,'analytical_convergence','Finite-volume errors against the independent analytical advection–diffusion eigenmode. The dashed line indicates first-order scaling anchored at the coarsest result; the finest reference is not compared with itself.','outputs/extended_verification_v5/analytical.json')
f,axs=canvas(8,'Baseline error evolution',cols=2,size=(9,3.8))
for ax,key,label in zip(axs,['mse','rmse'],['MSE ((kg/m²)²)','RMSE (kg/m²)']):ax.plot(t,errors[key],color=BLUE);ax.set(xlabel='Time (s)',ylabel=label);ax.ticklabel_format(axis='y',style='sci',scilimits=(0,0));ax.grid(alpha=.2)
save(8,f,'baseline_error_evolution','Time-resolved MSE and RMSE against the 800 × 92 reference restricted to output cells. '+BASE,SRC)
f,(ax,)=canvas(9,'Baseline signed error at 300 s');heat(ax,pred[-1]-ref[-1],symmetric=True);save(9,f,'baseline_signed_error','Signed hybrid prediction minus restricted FVM reference at t=300 s. '+BASE,SRC)
m0=float(initial_field(e).sum()*e.dx*e.dy)
f,(ax,)=canvas(10,'Flux-inclusive mass-budget error');ax.plot(t,100*mass['balance_error_kg']/m0,label='Hybrid PINN',color=BLUE);ax.plot(t,100*fine['balance_error_kg']/fine['mass_kg'][0],label='800 × 92 FVM',color=ORANGE);ax.axhline(0,c='gray',lw=.6);ax.set(xlabel='Time (s)',ylabel='[M(t) + exported mass − M₀] / M₀ (%)');ax.legend();save(10,f,'mass_budget','Flux-inclusive budget residual, normalized by each method’s prescribed-grid initial mass. Neural flux is integrated by the recorded output-time trapezoidal diagnostic; FVM export is integrated during internal steps. Neural maximum absolute budget error is 0.4483%.','Frozen checkpoint; simulation/validation_suite.py; simulation/finite_volume.py')
f,axs=canvas(11,'Adam component-loss diagnostics',cols=2,size=(9,3.8))
for key in ['data','pde','ic','mass']:axs[0].semilogy(steps,[v[key] for v in hist],label=key)
axs[0].set(xlabel='Adam iteration',ylabel='Unweighted component loss');axs[0].legend(ncol=2)
axs[1].semilogy(steps,[v['validation_mse'] for v in hist],color=GREEN);axs[1].set(xlabel='Adam iteration',ylabel='Normalized held-out label MSE');save(11,f,'component_loss_diagnostics','Unweighted Adam component losses and held-out label MSE for hybrid seed 2026. Validation labels are a split of the numerical label pool, not field observations or the finer evaluation reference. Boundary loss is omitted from the plot because its architectural enforcement keeps it near roundoff.','outputs/checkpoints/v5/hybrid_2026.history.json')
f,(ax,)=canvas(12,'Synthetic scenario geometry',size=(8.2,4.3))
for k,(name,parameters) in enumerate(TERRAIN.items()):ax.barh(k,parameters[0],color=[BLUE,GREEN,ORANGE,'#7D6098'][k]);ax.text(parameters[0]+4,k,f'{parameters[0]:g} m',va='center')
ax.set_yticks(range(4),labels=[s.title() for s in TERRAIN]);ax.set(xlabel='Prescribed channel width (m)',xlim=(0,405));ax.invert_yaxis();ax.set_title('All reaches: L = 1000 m; synthetic, not surveyed river locations',fontsize=10);save(12,f,'synthetic_scenario_geometry','Prescribed widths of the four synthetic terrain configurations. All reaches have length 1000 m. This replaces the old geographic-location map; no geographic calibration or surveyed site geometry is implied.','config.py:TERRAIN')
f,(ax,)=canvas(13,'Prescribed baseline initial condition');heat(ax,initial_field(e));save(13,f,'initial_condition','Prescribed compact initial pulse sampled at output-cell centres. It is independent of the neural prediction and common to both formulations. Initial support is 100<x<400 m, with transverse modulation; density is in kg/m².','config.py:initial_field')
vmax=max(float(pred[-1].max()),float(ref[-1].max()))
for n,field,slug,title in [(14,pred[-1],'neural_field','Hybrid PINN field at 300 s'),(15,ref[-1],'reference_field','FVM reference field at 300 s')]:
 f,(ax,)=canvas(n,title);heat(ax,field,vmax=vmax);save(n,f,slug,title+'. Identical axes and colour limits are used in Figs. 14 and 15. '+BASE,SRC)
f,(ax,)=canvas(16,'Relative field error over time');ax.plot(t,100*errors['relative_l2'],color=BLUE);ax.set(xlabel='Time (s)',ylabel='Per-time relative L2 error (%)');ax.grid(alpha=.2);save(16,f,'relative_error_evolution','Per-time relative L2 error of the baseline hybrid prediction. The final-time value is 2.1648%; the joint space–time relative L2 is 1.2318% and is a different statistic. This replaces the duplicated MSE figure.',SRC)
f,axs=canvas(17,'Signed error during transport',cols=3,size=(10,3.6));ids=[20,40,60];lim=max(np.abs(pred[i]-ref[i]).max() for i in ids)
for ax,i in zip(axs,ids):im=heat(ax,pred[i]-ref[i],title=f't = {t[i]:g} s',symmetric=True,vmax=lim,colorbar=False)
f.colorbar(im,ax=list(axs),label='Signed density error (kg/m²)',format='%.1e',shrink=.85)
save(17,f,'signed_error_snapshots','Baseline signed errors at 100, 200 and 300 s, with shared symmetric colour limits. Seed 2026; all axes are physical coordinates.',SRC)
f,axs=canvas(18,'Remaining mass and boundary export',cols=2,size=(9,3.8))
for ax,key,rkey,title in zip(axs,['mass_kg','integrated_outward_flux_kg'],['mass_kg','escaped_mass_kg'],['Remaining in-domain mass','Cumulative outward export']):ax.plot(t,mass[key],label='Hybrid PINN',color=BLUE);ax.plot(t,fine[rkey],'--',label='800 × 92 FVM',color=ORANGE);ax.set(xlabel='Time (s)',ylabel='Mass (kg)',title=title);ax.legend()
save(18,f,'mass_and_export','Remaining mass and cumulative outward boundary export for the baseline. Decreasing in-domain mass is expected as the pulse exits. Conservation is assessed through their sum and the initial mass, as in Fig. 10.','Frozen checkpoint; simulation/validation_suite.py; simulation/finite_volume.py')
f,(ax,)=canvas(19,'Measured CPU rollout time');names=['finite_volume','pinn'];means=[timing['summary'][n]['mean_s'] for n in names];sd=[timing['summary'][n]['sample_sd_s'] for n in names];ax.bar(['Output-grid FVM','Hybrid inference'],means,yerr=sd,capsize=6,color=[ORANGE,BLUE]);ax.set(ylabel='Wall-clock time (s)',ylim=(0,.125));ax.set_title('Mean ± sample SD; 10 repetitions; one warm-up',fontsize=10);save(19,f,'runtime_mean_sd','Measured CPU rollout time for equal 200 × 23 × 61 output workloads. AMD Ryzen 9 8940HX; two PyTorch threads; ten repetitions after one warm-up, alternating order. Training and loading are excluded. Neural inference is approximately 6.33 times slower.','outputs/verification_v5/timings.json:hybrid_2026_plain_normal')
f,axs=canvas(20,'Matched-formulation comparison',cols=3,size=(10,3.7));mx=max(ref[-1].max(),pred[-1].max(),data_pred[-1].max())
for ax,field,title in zip(axs,[ref[-1],pred[-1],data_pred[-1]],['FVM reference','Hybrid','Data-only']):im=heat(ax,field,title=title,vmax=mx,colorbar=False)
f.colorbar(im,ax=list(axs),label='Tracer density (kg/m²)',format='%.1e',shrink=.85)
save(20,f,'formulation_comparison','Baseline final fields at 300 s for the FVM reference, hybrid and data-only models, using shared colour limits and seed 2026. Both neural models share positivity/boundary architecture; hybrid training additionally uses physical and integrated-mass information. Three-seed statistics are reported in Table X.',SRC+'; outputs/checkpoints/v5/data_only_2026_final.pt')
mi=20
for n,field,slug,title,sym in [(21,mount['prediction'][mi],'mountain_wet_field','Mountain/wet prediction at 100 s',False),(22,mount['prediction'][mi]-mount['reference'][mi],'mountain_wet_error','Mountain/wet signed error at 100 s',True)]:
 f,(ax,)=canvas(n,title);heat(ax,field,exp=me,symmetric=sym);save(n,f,slug,title+'. Seed 2026; width 52.8 m. This snapshot is explicitly at 100 s, while reported scenario metrics cover the complete 0–300 s trajectory.','Frozen hybrid_2026_final.pt; evaluate(make_experiment("mountain","wet"))')
f,(ax,)=canvas(23,'Descriptive 90% residual-interval half-width');heat(ax,uq['half_width']);ax.collections[0].colorbar.set_label('Residual half-width (kg/m²)');save(23,f,'residual_half_width','Pointwise 90th-percentile absolute residual from the first 30 output times (0–145 s). Half-widths define descriptive numerical residual intervals, not guaranteed conformal intervals or field uncertainty. Evaluation on the remaining 31 times yields 72.5989% coverage.',SRC+'; simulation/uncertainty_quantification.py')
levels=np.array([.1,.2,.3,.4,.5,.6,.7,.8,.9,.95,.99]);coverage=[residual_interval_audit(pred,ref,float(v))['empirical_coverage'] for v in levels]
f,(ax,)=canvas(24,'Empirical residual-interval coverage');ax.plot([0,1],[0,1],'--',c='gray',label='Nominal equality');ax.plot(levels,coverage,'o-',c=BLUE,label='Chronological evaluation');ax.scatter([.9],[uq['empirical_coverage']],c=ORANGE,s=60,zorder=4);ax.annotate('90% nominal → 72.60% observed',(.9,uq['empirical_coverage']),xytext=(.3,.12),arrowprops=dict(arrowstyle='->',color=ORANGE),fontsize=10);ax.set(xlabel='Nominal residual quantile',ylabel='Empirical coverage',xlim=(0,1),ylim=(0,1));ax.legend(loc='upper left');save(24,f,'empirical_coverage','Empirical coverage on output times 150–300 s using half-widths estimated on 0–145 s. The diagonal is nominal equality, not a fitted calibration curve. Temporal dependence prevents claiming an exchangeability-based coverage guarantee.',SRC+'; simulation/uncertainty_quantification.py')
f,(ax,)=canvas(25,'Individual CPU timing repetitions')
for name,label,c in [('finite_volume','Output-grid FVM',ORANGE),('pinn','Hybrid inference',BLUE)]:ax.plot(range(1,11),[r[name] for r in timing['raw_seconds']],'o-',label=label,color=c)
ax.set(xlabel='Measured repetition',ylabel='Wall-clock time (s)',xticks=range(1,11));ax.legend();ax.grid(alpha=.2);save(25,f,'runtime_repetitions','All ten recorded baseline CPU timing repetitions under the protocol of Fig. 19. Execution order alternates; these are the original recorded timings, not new measurements. This is Fig. 25 in the revised document and replaces Fig. 26 in the original.','outputs/verification_v5/timings.json:hybrid_2026_plain_normal')
book.close()
assert len(records)==25 and sorted(r['figure'] for r in records)==list(range(1,26))
with (DATA/'baseline_error_timeseries.csv').open('w',newline='') as stream:
 w=csv.writer(stream);w.writerow(['time_s','mse','rmse','relative_l2']);w.writerows(zip(t,errors['mse'],errors['rmse'],errors['relative_l2']))
with (DATA/'coverage_curve.csv').open('w',newline='') as stream:
 w=csv.writer(stream);w.writerow(['nominal_level','empirical_coverage']);w.writerows(zip(levels,coverage))
hashes={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in [ROOT/SRC,ROOT/'outputs/verification_v5/metrics.json',ROOT/'outputs/verification_v5/timings.json',ROOT/'outputs/checkpoints/v5/hybrid_2026_final.pt',ROOT/'outputs/checkpoints/v5/data_only_2026_final.pt',ROOT/'config.py',ROOT/'core/pinn_model.py',ROOT/'simulation/validation_suite.py',ROOT/'simulation/finite_volume.py']}
(OUT/'figure_manifest.json').write_text(json.dumps({'figures':records,'source_sha256':hashes,'checked_against_saved_metrics':True,'retrained':False,'baseline_l2':errors['relative_l2_spacetime'],'baseline_budget':mass['max_relative_balance_error'],'baseline_coverage':uq['empirical_coverage'],'mountain_l2':mount['metrics']['relative_l2_spacetime']},indent=2),encoding='utf-8')
guide=['# Numbered replacement figures','Use Fig_01 through Fig_25 in the revised manuscript. Each image is supplied as a 300-dpi PNG for Word and a vector PDF for publication. Figures use current saved schema-5 results; no training was rerun.','## Insertion instructions','Replace the existing image in place, retain its available width, and paste the replacement caption below. Remove all old “legacy output” or “original schematic retained” notices. Review pagination after insertion; this package does not modify the manuscript.','Fig. 12 replaces the geographic map with synthetic scenario geometry: remove surrounding references to a retained geographic map. Fig. 13 now depicts the prescribed initial condition. Fig. 16 now shows relative L2 instead of duplicated MSE. Fig. 17 shows three error snapshots. Figs. 21–22 explicitly use t=100 s. Fig. 25 corresponds to Fig. 26 in the original manuscript.','These figures use the original upwind-reference comparison. Pair them with the matching audited manuscript captions; refined-reference results are supplied separately as Figures 26-28.','## Replacement captions']
for r in records:guide += [f'### Fig. {r["figure"]} — {r["file"]}',r['caption'],f'Source: {r["source"]}']
(OUT/'INSERTION_GUIDE.md').write_text('\n\n'.join(guide),encoding='utf-8')
print('Finished 25 numbered figures and manifest.',flush=True)
