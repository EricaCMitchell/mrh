import sys
sys.path.append ('../../../../')
from pyscf import gto, mcscf, scf, fci, dft
from mrh.my_pyscf.mcpdft import mcpdft, otfnal

''' C atom triplet-singlet gap reported in JCTC 2014, 10, 3669
    CASSCF(4,4):    1.6 eV
    tPBE:           1.1 eV
    tBLYP:          1.0 eV
    'Vertical' means triplet orbitals for singlet state
    'Relaxed' means optimized orbitals for both states
'''
file1 = open("Catom31gap.out","a+")

mol = gto.M (atom = 'C 0 0 0', basis='cc-pvtz', spin = 2, symmetry=True)
mf = scf.RHF (mol)
mf.kernel ()
hs = mcscf.CASSCF (mf, 4, (3, 1))
ehs = hs.kernel ()[0]

mf.mo_coeff = hs.mo_coeff
ls_vert = mcscf.CASCI (mf, 4, (2,2))
ls_vert.fcisolver = fci.solver (mf.mol, singlet=True)
els_vert = ls_vert.kernel ()[0]

ls_rel = mcscf.CASSCF (mf, 4, (2,2))
ls_rel.fcisolver = fci.solver (mf.mol, singlet=True)
els_rel = ls_rel.kernel ()[0]

file1.write("CASSCF high-spin energy: {:.8f}".format (ehs)+"\n")
file1.write("CASSCF (vertical) low-spin energy: {:.8f}".format (els_vert)+"\n")
file1.write("CASSCF (relaxed) low-spin energy: {:.8f}".format (els_rel)+"\n")
file1.write("CASSCF vertical excitation energy (eV): {:.8f}".format (27.2114 * (els_vert - ehs))+"\n")
file1.write("CASSCF relaxed excitation energy (eV): {:.8f}".format (27.2114 * (els_rel - ehs))+"\n")
print ("CASSCF(4,4) Completed")

ks = dft.UKS (mol)
ks.xc = 'pbe'
ks.grids.level = 9
ot = otfnal.transfnal (ks)

els_vert = mcpdft.kernel (ls_vert, ot)[0]
els_rel = mcpdft.kernel (ls_rel, ot)[0]
ehs = mcpdft.kernel (hs, ot)[0]
file1.write("MC-PDFT (tPBE) high-spin energy: {:.8f}".format (ehs)+"\n")
file1.write("MC-PDFT (tPBE) (vertical) low-spin energy: {:.8f}".format (els_vert)+"\n")
file1.write("MC-PDFT (tPBE) (relaxed) low-spin energy: {:.8f}".format (els_rel)+"\n")
file1.write("MC-PDFT (tPBE) vertical excitation energy (eV): {:.8f}".format (27.2114 * (els_vert - ehs))+"\n")
file1.write("MC-PDFT (tPBE) relaxed excitation energy (eV): {:.8f}".format (27.2114 * (els_rel - ehs))+"\n")
print ("MC-PDFT (tPBE) Completed")

ks = dft.UKS (mol)
ks.xc = 'blyp'
#ks.grids.level = 9
ot = otfnal.transfnal (ks)

els_vert = mcpdft.kernel (ls_vert, ot)[0]
els_rel = mcpdft.kernel (ls_rel, ot)[0]
ehs = mcpdft.kernel (hs, ot)[0]
file1.write("MC-PDFT (tBLYP) high-spin energy: {:.8f}".format (ehs)+"\n")
file1.write("MC-PDFT (tBLYP) (vertical) low-spin energy: {:.8f}".format (els_vert)+"\n")
file1.write("MC-PDFT (tBLYP) (relaxed) low-spin energy: {:.8f}".format (els_rel)+"\n")
file1.write("MC-PDFT (tBLYP) vertical excitation energy (eV): {:.8f}".format (27.2114 * (els_vert - ehs))+"\n")
file1.write("MC-PDFT (tBLYP) relaxed excitation energy (eV): {:.8f}".format (27.2114 * (els_rel - ehs)))
print ("MC-PDFT (tBLYP) Completed")

file1.close()
