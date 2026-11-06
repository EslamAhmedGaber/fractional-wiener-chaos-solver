
import sys,time,numpy as np
sys.path.insert(0,r"D:\PAPERS_HUB\Paper_VIII\INDEPENDENT_RL_FOURIER_BASIS_20260925\TWO_PAPER_SPLIT_20261027\04_SUBMISSION_PACKAGES\CSF_ROUND_20261105\working\evidence\round3_fwhe\solver_v1.1.3"); import hwc
P,Eq=hwc.Poly,hwc.Equation
mem=Eq(1.0,P([[(0.0198+0.02**2/2,(1,))]]),P([[(0.02,(1,))]]),memory=dict(a=0.5,alpha=0.5))
tl=np.array([4.,8.,12.,16.,20.])
t0=time.perf_counter(); r=hwc.solve(mem,0.7,20.0,33,1,tl,form='correlated',memory_opts={'ngrid':401}); print(time.perf_counter()-t0)
np.save(r"D:\PAPERS_HUB\Paper_VIII\INDEPENDENT_RL_FOURIER_BASIS_20260925\TWO_PAPER_SPLIT_20261027\04_SUBMISSION_PACKAGES\CSF_ROUND_20261105\working\evidence\round3_fwhe\_v113_var.npy",r.variance()); np.save(r"D:\PAPERS_HUB\Paper_VIII\INDEPENDENT_RL_FOURIER_BASIS_20260925\TWO_PAPER_SPLIT_20261027\04_SUBMISSION_PACKAGES\CSF_ROUND_20261105\working\evidence\round3_fwhe\_v113_coef.npy",r.coef)
