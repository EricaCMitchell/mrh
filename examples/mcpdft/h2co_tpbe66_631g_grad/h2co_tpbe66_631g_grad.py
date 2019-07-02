import sys
sys.path.append ('../../../../')
from pyscf import gto, scf, mcscf
from pyscf.lib import logger
from pyscf.data.nist import BOHR
from mrh.my_pyscf import mcpdft
from mrh.my_pyscf.fci import csf_solver
from scipy import linalg
import numpy as np
import math
from time import process_time

t0 = process_time()
file1 = open("grad_example.out","w")

# Energy calculation
h2co_casscf66_631g_xyz = '''C  0.534004  0.000000  0.000000
O -0.676110  0.000000  0.000000
H  1.102430  0.000000  0.920125
H  1.102430  0.000000 -0.920125'''
mol = gto.M (atom = h2co_casscf66_631g_xyz, basis = '6-31g', symmetry = False, verbose = logger.INFO, output = 'h2co_tpbe66_631g_grad.log')
mf = scf.RHF (mol).run ()
mc = mcpdft.CASSCF (mf, 'tPBE', 6, 6, grids_level=9)
mc.fcisolver = csf_solver (mol, smult = 1)
mc.kernel ()
t1 = process_time()
print("Energy Calc Complete")

# Gradient calculation
mc_grad = mc.nuc_grad_method ()
dE = mc_grad.kernel ()
file1.write("tPBE(6,6)/6-31g gradient of formaldehyde at the CASSCF(6,6)/6-31g geometry:\n")
for ix, row in enumerate (dE):
    file1.write("{:1s} {:11.8f} {:11.8f} {:11.8f}".format (mol.atom_pure_symbol (ix), *row)+"\n")
file1.write("\nOpenMolcas's opinions (note that there is no way to use the same grid in PySCF and OpenMolcas)\n")
file1.write(("Analytical implementation:\n"
"C -0.02790613  0.00000000 -0.00000000\n"
"O  0.05293225 -0.00000000  0.00000000\n"
"H -0.01251306 -0.00000000 -0.01568837\n"
"H -0.01251306 -0.00000000  0.01568837\n"+"\n"+
"Numerical algorithm:\n"
"C -0.028019    0.000000   -0.000000\n"
"O  0.053037   -0.000000    0.000000\n"
"H -0.012509   -0.000000   -0.015737\n"
"H -0.012509   -0.000000    0.015737\n"))
file1.close()
t2 = process_time()

print ("Energy Calculation: ", t1-t0, " seconds")
print ("Gradient Calculation: ", t2-t1, " seconds")
print ("Job has run successfully... Fare thee well") 
