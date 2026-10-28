from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"evidence/solver_snapshot/code"))
import numpy as np
import hwc
P, Eq = hwc.Poly, hwc.Equation
ou=Eq(0.,P([[(-1.,(1,))]]),P([[(1.,(0,))]]),name="OU")
times=np.array([.5,1.,2.])
out=hwc.solve(ou,.75,2.,17,1,times)
print("times:",times)
print("means:",out.mean[:,0])
print("variances:",out.variance()[:,0])
