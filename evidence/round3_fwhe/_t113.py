"""Caputo-memory timing probe using the versioned solver snapshot."""
from pathlib import Path
HERE=Path(__file__).resolve().parent

import sys,time,numpy as np
sys.path.insert(0,str(HERE/'solver_v1.1.3')); import hwc
P,Eq=hwc.Poly,hwc.Equation
mem=Eq(1.0,P([[(0.0198+0.02**2/2,(1,))]]),P([[(0.02,(1,))]]),memory=dict(a=0.5,alpha=0.5))
tl=np.array([4.,8.,12.,16.,20.])
t0=time.perf_counter(); r=hwc.solve(mem,0.7,20.0,33,1,tl,form='correlated',memory_opts={'ngrid':401}); print(time.perf_counter()-t0)
np.save(HERE/'_v113_var.npy',r.variance()); np.save(HERE/'_v113_coef.npy',r.coef)
