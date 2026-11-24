"""Audit local raw datasets without changing the trained experiment.

Run in the optional data-audit environment; raw data are not redistributed.
"""
from pathlib import Path
import json, hashlib, sys
import numpy as np
import shapefile
import netCDF4
import rasterio
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from config import TERRAIN, SEASON_DISCHARGE

def digest(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda:f.read(8*1024*1024),b''):h.update(block)
    return h.hexdigest()

def run():
    out=ROOT/'outputs/data_provenance';out.mkdir(parents=True,exist_ok=True)
    files=[]
    for p in sorted((ROOT/'data').rglob('*')):
        if p.is_file() and p.suffix in ['.shp','.dbf','.shx','.prj','.nc','.tif']:
            files.append({'path':p.relative_to(ROOT).as_posix(),'bytes':p.stat().st_size,'sha256':digest(p)})
            print('Hashed',p.name,flush=True)
    reader=shapefile.Reader(str(ROOT/'data/grwl/GRWL_summaryStats.shp'))
    fields=['width_min_','width_med_','width_mean','width_max_']
    values=np.array([[r[n] for n in fields] for r in reader.iterRecords()],dtype=float)
    stats=dict(zip(fields,np.nanmedian(values,axis=0).tolist()))
    lo,med,mean,hi=[stats[n] for n in fields]
    ranges={'mountain':[round(lo,1),round(med*.7,1)],'plain':[round(med*.7,1),round(med*1.5,1)],'urban':[round(med*.5,1),round(med,1)],'coastal':[round(med*1.5,1),round(hi,1)]}
    widths={k:sum(v)/2 for k,v in ranges.items()}
    assert all(np.isclose(widths[k],TERRAIN[k][0],rtol=0,atol=1e-9) for k in widths)
    with netCDF4.Dataset(ROOT/'data/chirps/chirps-v2.0.2022.monthly.nc') as nc:
        chirps={'dimensions':{k:len(v) for k,v in nc.dimensions.items()},'variables':list(nc.variables),'attributes':{k:str(nc.getncattr(k)) for k in nc.ncattrs()},'precip_units':str(getattr(nc.variables['precip'],'units','not recorded'))}
    hydro=[]
    for p in sorted((ROOT/'data/hydrosheds').rglob('*.tif')):
        with rasterio.open(p) as r:
            hydro.append({'path':p.relative_to(ROOT).as_posix(),'crs':str(r.crs),'shape':[r.height,r.width],'resolution':list(r.res),'nodata':r.nodata,'dtype':r.dtypes[0]})
    report={'raw_files':files,'grwl':{'records':len(values),'median_fields_m':stats,'derived_ranges_m':ranges,'midpoints_m':widths,'matches_config':True,'interpretation':'Dataset-informed global width abstraction; terrain classes and multipliers are modelling assumptions, not observed terrain classifications or a geolocated reach.'},'chirps':chirps,'stored_chirps_factors':json.loads((ROOT/'data_validation/outputs/climate_scaling.json').read_text()),'active_discharge_factors':SEASON_DISCHARGE,'hydrosheds':hydro,'era5':{'found':False,'search_scope':'project data, data_validation, outputs and active source references'},'conclusions':['GRWL global summary medians reproduce the configured width midpoints.','CHIRPS precipitation factors are not the active discharge multipliers; no rainfall-runoff calibration is implemented.','HydroSHEDS direction and accumulation rasters are present but no extraction feeds the reported slope or discharge. Those values are prescribed assumptions.','No ERA5 input or implemented ingestion is evidenced.','None of these resources supplies observed tracer-density labels or field validation.']}
    (out/'audit.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    print(json.dumps(report['grwl'],indent=2),flush=True)
if __name__=='__main__':run()
