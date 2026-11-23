"""Independent second-order reference for the prescribed separable tracer problem.

Centered conservative longitudinal face fluxes + Crank-Nicolson; transverse
Neumann cosine evolves analytically. This exploits constant coefficients and the
specific IC; it is not a general river solver or an exact neural constraint.
"""
import numpy as np
from scipy.sparse import diags,eye
from scipy.sparse.linalg import splu

def longitudinal(exp,nx=6400,dt_max=0.1,times=None,initial_function=None):
    times=np.asarray(exp.times() if times is None else times,dtype=float)
    if type(nx) is not int or nx<2 or not np.isfinite(dt_max) or dt_max<=0:
        raise ValueError('Invalid grid or timestep')
    if times.ndim!=1 or len(times)<2 or times[0]!=0 or np.any(np.diff(times)<=0) or times[-1]>exp.horizon:
        raise ValueError('Times must start at zero and increase within horizon')
    dx=exp.length/nx;u=exp.u;D=exp.diffusion
    if u*dx>2*D+1e-14:
        raise ValueError('Centered reference needs cell Peclet <=2; refine the grid')
    q,w=np.polynomial.legendre.leggauss(6)
    points=(np.arange(nx)[:,None]+.5+.5*q)*dx
    if initial_function is None:
        r=(points/exp.length-.25)/.15;inside=np.abs(r)<1
        raw=np.where(inside,np.exp(1-1/np.where(inside,1-r*r,1)),0)
    else:raw=initial_function(points)
    c=raw@(w/2)
    lo=np.full(nx-1,D/dx**2+u/(2*dx));hi=np.full(nx-1,D/dx**2-u/(2*dx))
    mid=np.full(nx,-2*D/dx**2);mid[0]=-3*D/dx**2-u/(2*dx);mid[-1]=-D/dx**2-u/(2*dx)
    A=diags([lo,mid,hi],[-1,0,1],format='csc');I=eye(nx,format='csc')
    cache={};frames=[c.copy()];mass=[c.sum()*dx];export=[0.];total=0.
    for interval in np.diff(times):
        steps=int(np.ceil(interval/dt_max));dt=float(interval/steps)
        if dt not in cache:
            cache[dt]=(splu(I-.5*dt*A),I+.5*dt*A)
        factor,right=cache[dt]
        for _ in range(steps):
            old_flux=u*c[-1]+2*D*c[0]/dx
            new=factor.solve(right@c)
            total+=.5*dt*(old_flux+u*new[-1]+2*D*new[0]/dx);c=new
        frames.append(c.copy());mass.append(c.sum()*dx);export.append(total)
    mass=np.asarray(mass);export=np.asarray(export);frames=np.asarray(frames)
    return dict(x=(np.arange(nx)+.5)*dx,times=times,frames=frames,mass=mass,export=export,
                relative_balance=float(np.max(np.abs(mass+export-mass[0]))/abs(mass[0])),minimum=float(frames.min()),nx=nx,dt_max=dt_max)

def solve_separable(exp,nx=6400,dt_max=.1,times=None):
    if nx%exp.nx:raise ValueError('Reference nx must be divisible by output nx')
    result=longitudinal(exp,nx,dt_max,times)
    a=result['frames'].reshape(len(result['times']),exp.nx,nx//exp.nx).mean(2)
    y=exp.grid()[1]
    # Exact average of cosine over each transverse output cell.
    lateral=np.cos(2*np.pi*y/exp.width)*np.sinc(1/exp.ny)
    decay=np.exp(-exp.diffusion*(2*np.pi/exp.width)**2*result['times'])
    result['output_frames']=exp.initial_density*a[:,:,None]*(1+.3*decay[:,None,None]*lateral[None,None,:])
    return result
