"""Compare the fresh verification to archived scientific values; timing may vary."""
from pathlib import Path
import json,hashlib
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
out=ROOT/'outputs/final_audit_v031'
old=json.loads((ROOT/'outputs/verification_v5/metrics.json').read_text())
new=json.loads((out/'metrics.json').read_text())
key=lambda r:(r['variant'],r['seed'],r['terrain'],r['season'])
a={key(r):r for r in old};b={key(r):r for r in new};assert set(a)==set(b) and len(b)==72
fields=['relative_l2_spacetime','mass_balance_relative_error','initial_relative_l2','minimum_density_kg_m2']
differences={f:max(abs(a[k][f]-b[k][f]) for k in a) for f in fields}
assert max(differences.values())<1e-10,differences
assert all(r['accepted_numerically'] for r in new)
oldfields=np.load(ROOT/'outputs/verification_v5/baseline_fields.npz');newfields=np.load(out/'baseline_fields.npz')
array_difference=max(float(np.max(np.abs(oldfields[k]-newfields[k]))) for k in ['prediction','reference'])
assert array_difference<1e-14
report={'cases':len(new),'passed':sum(r['accepted_numerically'] for r in new),'max_metric_absolute_differences':differences,'max_baseline_array_difference':array_difference,'timing_policy':'Fresh timings are retained separately; manuscript timings remain the declared original ten-repeat experiment, not selected from reruns.','checkpoints':'No training or checkpoint replacement in this audit.'}
(out/'reproduction_comparison.json').write_text(json.dumps(report,indent=2))
print(json.dumps(report,indent=2))
