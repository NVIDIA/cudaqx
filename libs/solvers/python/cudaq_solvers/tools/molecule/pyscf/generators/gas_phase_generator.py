# ============================================================================ #
# Copyright (c) 2024 NVIDIA Corporation & Affiliates.                          #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #

from ..hamiltonian_generator import HamiltonianGenerator
import numpy as np
import json
from functools import reduce
try:
    from pyscf import gto, scf, cc, ao2mo, mp, mcscf, solvent, fci
except RuntimeError:
    print(
        'PySCF should be installed to use cudaq-pyscf tool. Use pip install pyscf'
    )


class GasPhaseGenerator(HamiltonianGenerator):

    def name(self):
        return 'gas_phase'

    def generate_molecular_spin_ham_restricted(self, h1e, h2e, ecore):

        # This function generates the molecular spin Hamiltonian
        # H= E_core+sum_{pq}  h_{pq} a_p^dagger a_q +
        #                          0.5 * h_{pqrs} a_p^dagger a_q^dagger a_r a_s
        # h1e: one body integrals h_{pq}
        # h2e: two body integrals h_{pqrs}
        # ecore: constant (nuclear repulsion or core energy in the active space Hamiltonian)

        # Total number of qubits equals the number of spin molecular orbitals
        nqubits = 2 * h1e.shape[0]

        # Initialization
        one_body_coeff = np.zeros((nqubits, nqubits))
        two_body_coeff = np.zeros((nqubits, nqubits, nqubits, nqubits))

        for p in range(nqubits // 2):
            for q in range(nqubits // 2):

                # p & q have the same spin <a|a>= <b|b>=1
                # <a|b>=<b|a>=0 (orthogonal)
                one_body_coeff[2 * p, 2 * q] = h1e[p, q]
                one_body_coeff[2 * p + 1, 2 * q + 1] = h1e[p, q]

                for r in range(nqubits // 2):
                    for s in range(nqubits // 2):

                        # Same spin (aaaa, bbbbb) <a|a><a|a>, <b|b><b|b>
                        two_body_coeff[2 * p, 2 * q, 2 * r,
                                       2 * s] = 0.5 * h2e[p, q, r, s]
                        two_body_coeff[2 * p + 1, 2 * q + 1, 2 * r + 1,
                                       2 * s + 1] = 0.5 * h2e[p, q, r, s]

                        # Mixed spin(abab, baba) <a|a><b|b>, <b|b><a|a>
                        #<a|b>= 0 (orthogoanl)
                        two_body_coeff[2 * p, 2 * q + 1, 2 * r + 1,
                                       2 * s] = 0.5 * h2e[p, q, r, s]
                        two_body_coeff[2 * p + 1, 2 * q, 2 * r,
                                       2 * s + 1] = 0.5 * h2e[p, q, r, s]

        return one_body_coeff, two_body_coeff, ecore

    def generate_molecular_spin_ham_ur(self, h1e, h2e, h2e_prime, ecore):

        # This function generates the molecular spin Hamiltonian
        # H= E_core+sum_{pq}  h_{pq} a_p^dagger a_q +
        #                          0.5 * h_{pqrs} a_p^dagger a_q^dagger a_r a_s
        # h1e: one body integrals h_{pq}
        # h2e: two body integrals h_{pqrs}
        # ecore: constant (nuclear repulsion or core energy in the active space Hamiltonian)

        # Total number of qubits equals the number of spin molecular orbitals
        nqubits = 2 * h1e[0].shape[0]

        # Initialization
        one_body_coeff = np.zeros((nqubits, nqubits))
        two_body_coeff = np.zeros((nqubits, nqubits, nqubits, nqubits))

        for p in range(nqubits // 2):
            for q in range(nqubits // 2):

                # p & q have the same spin <a|a>, <b,b>
                one_body_coeff[2 * p, 2 * q] = h1e[0, p, q]
                one_body_coeff[2 * p + 1, 2 * q + 1] = h1e[1, p, q]

                for r in range(nqubits // 2):
                    for s in range(nqubits // 2):

                        # Same spin (aaaa, bbbbb) <a|a><a|a>, <b|b><b|b>
                        two_body_coeff[2 * p, 2 * q, 2 * r,
                                       2 * s] = 0.5 * h2e[0, p, q, r, s]
                        two_body_coeff[2 * p + 1, 2 * q + 1, 2 * r + 1,
                                       2 * s + 1] = 0.5 * h2e[2, p, q, r, s]

                        # Mixed spin(abba, baab) <a|a><b|b>, <b|b><a|a>
                        two_body_coeff[2 * p + 1, 2 * q, 2 * r,
                                       2 * s + 1] = 0.5 * h2e_prime[p, q, r, s]
                        two_body_coeff[2 * p, 2 * q + 1, 2 * r + 1,
                                       2 * s] = 0.5 * h2e[1, p, q, r, s]

        return one_body_coeff, two_body_coeff, ecore


####################################################################

# A- Without solvent
#############################
## Beginning of simulation
#############################

    def get_spin_hamiltonian(self, xyz:str, spin:int, charge: int, basis:str, symmetry:bool=False, memory:float=4000,cycles:int=100, \
                                initguess:str='minao', UR:bool=False, nele_cas=None, norb_cas=None, MP2:bool=False, natorb:bool=False,\
                                casci:bool=False, ccsd:bool=False, casscf:bool=False,integrals_natorb:bool=False, \
                                    integrals_casscf:bool=False, verbose:bool=False, cache_data=True, outFileName=None):
        ################################
        # Initialize the molecule
        ################################
        filename = xyz.split('.')[0] if outFileName == None else outFileName
        energies = {}

        if (nele_cas is None) and (norb_cas is not None):
            raise RuntimeError(
                "WARN: nele_cas is None and norb_cas is not None. nele_cas and norb_cas should be either both None\
                        or have values")

        if (nele_cas is not None) and (norb_cas is None):
            raise RuntimeError(
                "WARN: nele_cas is not None and norb_cas is None. nele_cas and norb_cas should be either both None\
                        or have values")

        ########################################################################
        # To add (coming soon)

        if UR and nele_cas is None:
            raise RuntimeError(
                "WARN: Unrestricted spin calculation for the full space is not supported yet on Cudaq.\
                        Only active space is currently supported for the unrestricted spin calculations."
            )

        mol = gto.M(atom=xyz,
                    spin=spin,
                    charge=charge,
                    basis=basis,
                    max_memory=memory,
                    symmetry=symmetry,
                    output=filename + '-pyscf.log',
                    verbose=4)

        ##################################
        # Mean field (HF)
        ##################################

        if UR:
            myhf = scf.UHF(mol)
            myhf.max_cycle = cycles
            myhf.chkfile = filename + '-pyscf.chk'
            myhf.init_guess = initguess
            myhf.kernel()

            norb = myhf.mo_coeff[0].shape[1]
            if verbose:
                print('[pyscf] Total number of alpha molecular orbitals = ',
                      norb)
            norb = myhf.mo_coeff[1].shape[1]
            if verbose:
                print('[pyscf] Total number of beta molecular orbitals = ',
                      norb)

        else:
            myhf = scf.RHF(mol)
            myhf.max_cycle = cycles
            myhf.chkfile = filename + '-pyscf.chk'
            myhf.init_guess = initguess
            myhf.kernel()

            norb = myhf.mo_coeff.shape[1]
            if verbose:
                print('[pyscf] Total number of orbitals = ', norb)

        nelec = mol.nelectron
        energies['hf_energy'] = myhf.e_tot
        if verbose:
            print('[pyscf] Total number of electrons = ', nelec)
            print('[pyscf] HF energy = ', myhf.e_tot)

        ##########################
        # MP2
        ##########################
        if MP2:

            if UR:
                mymp = mp.UMP2(myhf)
                mp_ecorr, mp_t2 = mymp.kernel()
                if verbose:
                    print('[pyscf] UR-MP2 energy= ', mymp.e_tot)

                if integrals_natorb or natorb:
                    # Compute natural orbitals
                    dma, dmb = mymp.make_rdm1()
                    noon_a, U_a = np.linalg.eigh(dma)
                    noon_b, U_b = np.linalg.eigh(dmb)
                    noon_a = np.flip(noon_a)
                    noon_b = np.flip(noon_b)

                    if verbose:
                        print(
                            '[pyscf] Natural orbital (alpha orbitals) occupation number from UR-MP2: '
                        )
                        print(noon_a)
                        print(
                            '[pyscf] Natural orbital (beta orbitals) occupation number from UR-MP2: '
                        )
                        print(noon_b)

                    natorbs = np.zeros(np.shape(myhf.mo_coeff))
                    natorbs[0, :, :] = np.dot(myhf.mo_coeff[0], U_a)
                    natorbs[0, :, :] = np.fliplr(natorbs[0, :, :])
                    natorbs[1, :, :] = np.dot(myhf.mo_coeff[1], U_b)
                    natorbs[1, :, :] = np.fliplr(natorbs[1, :, :])

            else:
                if spin != 0:
                    raise RuntimeError("WARN: ROMP2 is unvailable in pyscf.")
                else:
                    mymp = mp.MP2(myhf)
                    mp_ecorr, mp_t2 = mymp.kernel()
                    if verbose:
                        print('[pyscf] R-MP2 energy= ', mymp.e_tot)

                    if integrals_natorb or natorb:
                        # Compute natural orbitals
                        noons, natorbs = mcscf.addons.make_natural_orbitals(
                            mymp)
                        if verbose:
                            print(
                                '[pyscf] Natural orbital occupation number from R-MP2: '
                            )
                            print(noons)

        #######################################
        # CASCI if active space is defined
        # FCI if the active space is None
        ######################################
        if casci:

            if UR:
                if natorb:
                    mycasci = mcscf.UCASCI(myhf, norb_cas, nele_cas)
                    mycasci.kernel(natorbs)
                    if verbose:
                        print(
                            '[pyscf] UR-CASCI energy using natural orbitals= ',
                            mycasci.e_tot)
                    energies['UR-CASCI'] = mycasci.e_tot
                else:
                    mycasci_mo = mcscf.UCASCI(myhf, norb_cas, nele_cas)
                    mycasci_mo.kernel()
                    if verbose:
                        print(
                            '[pyscf] UR-CASCI energy using molecular orbitals= ',
                            mycasci_mo.e_tot)
                    energies['UR-CASCI'] = mycasci_mo.e_tot

            else:
                if nele_cas is None:
                    myfci = fci.FCI(myhf)
                    result = myfci.kernel()
                    if verbose:
                        print('[pyscf] FCI energy = ', result[0])
                    energies['fci_energy'] = result[0]

                else:
                    if natorb and (spin == 0):
                        mycasci = mcscf.CASCI(myhf, norb_cas, nele_cas)
                        mycasci.kernel(natorbs)
                        if verbose:
                            print(
                                '[pyscf] R-CASCI energy using natural orbitals= ',
                                mycasci.e_tot)

                        energies['R-CASCI'] = mycasci.e_tot

                    elif natorb and (spin != 0):
                        raise RuntimeError(
                            "WARN: Natural orbitals cannot be computed. ROMP2 is unvailable in pyscf."
                        )

                    else:
                        mycasci_mo = mcscf.CASCI(myhf, norb_cas, nele_cas)
                        mycasci_mo.kernel()
                        if verbose:
                            print(
                                '[pyscf] R-CASCI energy using molecular orbitals= ',
                                mycasci_mo.e_tot)

                        energies['R-CASCI'] = mycasci_mo.e_tot

        ########################
        # CCSD
        ########################
        if ccsd:

            if UR:

                mc = mcscf.UCASCI(myhf, norb_cas, nele_cas)
                frozen = []
                frozen = [y for y in range(0, mc.ncore[0])]
                frozen += [
                    y for y in range(mc.ncore[0] +
                                     mc.ncas, len(myhf.mo_coeff[0]))
                ]

                if natorb:
                    mycc = cc.UCCSD(myhf, frozen=frozen, mo_coeff=natorbs)
                    mycc.max_cycle = cycles
                    mycc.kernel()
                    if verbose:
                        print(
                            '[pyscf] UR-CCSD energy of the active space using natural orbitals= ',
                            mycc.e_tot)

                else:
                    mycc = cc.UCCSD(myhf, frozen=frozen)
                    mycc.max_cycle = cycles
                    mycc.kernel()
                    if verbose:
                        print(
                            '[pyscf] UR-CCSD energy of the active space using molecular orbitals= ',
                            mycc.e_tot)

                energies['UR-CCSD'] = mycc.e_tot

            else:
                if nele_cas is None:
                    mycc = cc.CCSD(myhf)
                    mycc.max_cycle = cycles
                    mycc.kernel()
                    if verbose:
                        print('[pyscf] Total R-CCSD energy = ', mycc.e_tot)

                else:
                    mc = mcscf.CASCI(myhf, norb_cas, nele_cas)
                    frozen = []
                    frozen += [y for y in range(0, mc.ncore)]
                    frozen += [
                        y for y in range(mc.ncore +
                                         norb_cas, len(myhf.mo_coeff))
                    ]
                    if natorb and (spin == 0):
                        mycc = cc.CCSD(myhf, frozen=frozen, mo_coeff=natorbs)
                        mycc.max_cycle = cycles
                        mycc.kernel()
                        if verbose:
                            print(
                                '[pyscf] R-CCSD energy of the active space using natural orbitals= ',
                                mycc.e_tot)

                    elif natorb and (spin != 0):
                        raise RuntimeError(
                            "WARN: Natural orbitals cannot be computed. ROMP2 is unvailable in pyscf."
                        )

                    else:
                        mycc = cc.CCSD(myhf, frozen=frozen)
                        mycc.max_cycle = cycles
                        mycc.kernel()
                        if verbose:
                            print(
                                '[pyscf] R-CCSD energy of the active space using molecular orbitals= ',
                                mycc.e_tot)

                    energies['R-CCSD'] = mycc.e_tot

        #########################
        # CASSCF
        #########################
        if casscf:
            if nele_cas is None:
                raise RuntimeError("WARN: You should define the active space.")

            if UR:
                if natorb:
                    mycas = mcscf.UCASSCF(myhf, norb_cas, nele_cas)
                    mycas.max_cycle_macro = cycles
                    mycas.kernel(natorbs)
                    if verbose:
                        print(
                            '[pyscf] UR-CASSCF energy using natural orbitals= ',
                            mycas.e_tot)
                else:
                    mycas = mcscf.UCASSCF(myhf, norb_cas, nele_cas)
                    mycas.max_cycle_macro = cycles
                    mycas.kernel()
                    if verbose:
                        print(
                            '[pyscf] UR-CASSCF energy using molecular orbitals= ',
                            mycas.e_tot)

                energies['UR-CASSCF'] = mycas.e_tot

            else:

                if natorb and (spin == 0):
                    mycas = mcscf.CASSCF(myhf, norb_cas, nele_cas)
                    mycas.max_cycle_macro = cycles
                    mycas.kernel(natorbs)
                    if verbose:
                        print(
                            '[pyscf] R-CASSCF energy using natural orbitals= ',
                            mycas.e_tot)

                elif natorb and (spin != 0):
                    raise RuntimeError(
                        "WARN: Natural orbitals cannot be computed. ROMP2 is unvailable in pyscf."
                    )

                else:
                    mycas = mcscf.CASSCF(myhf, norb_cas, nele_cas)
                    mycas.max_cycle_macro = cycles
                    mycas.kernel()
                    if verbose:
                        print(
                            '[pyscf] R-CASSCF energy using molecular orbitals= ',
                            mycas.e_tot)

                energies['R-CASSCF'] = mycas.e_tot

        ###################################
        # CASCI: FCI of the active space
        ##################################
        if casci and casscf:

            if UR:
                h1e_cas, ecore = mycas.get_h1eff()
                h2e_cas = mycas.get_h2eff()

                e_fci, fcivec = fci.direct_uhf.kernel(h1e_cas,
                                                      h2e_cas,
                                                      norb_cas,
                                                      nele_cas,
                                                      ecore=ecore)
                if verbose:
                    print('[pyscf] UR-CASCI energy using the casscf orbitals= ',
                          e_fci)

                energies['UR-CASCI'] = e_fci

            else:
                if natorb and (spin != 0):
                    raise RuntimeError(
                        "WARN: Natural orbitals cannot be computed. ROMP2 is unavailable in pyscf."
                    )

                h1e_cas, ecore = mycas.get_h1eff()
                h2e_cas = mycas.get_h2eff()

                e_fci, fcivec = fci.direct_spin1.kernel(h1e_cas,
                                                        h2e_cas,
                                                        norb_cas,
                                                        nele_cas,
                                                        ecore=ecore)
                if verbose:
                    print('[pyscf] R-CASCI energy using the casscf orbitals= ',
                          e_fci)

                energies['R-CASCI'] = e_fci

        ###################################################################################
        # Computation of one- and two- electron integrals for the active space Hamiltonian
        ###################################################################################

        if UR:
            if integrals_natorb:
                mc = mcscf.UCASCI(myhf, norb_cas, nele_cas)
                h1e, ecore = mc.get_h1eff(natorbs)
                h1e_cas = np.array(h1e)
                h2e = mc.get_h2eff(natorbs)
                h2e_cas = np.array(h2e)
                h2e_cas[0] = np.asarray(h2e_cas[0].transpose(0, 2, 3, 1),
                                        order='C')
                h2e_cas[1] = np.asarray(h2e_cas[1].transpose(0, 2, 3, 1),
                                        order='C')
                h2e_cas[2] = np.asarray(h2e_cas[2].transpose(0, 2, 3, 1),
                                        order='C')
                h2e_cas_prime = np.asarray(h2e_cas[1].transpose(2, 0, 1, 3),
                                           order='C')

            elif integrals_casscf:
                if casscf:
                    h1e, ecore = mycas.get_h1eff()
                    h1e_cas = np.array(h1e)
                    h2e = mycas.get_h2eff()
                    h2e_cas = np.array(h2e)
                    h2e_cas[0] = np.asarray(h2e_cas[0].transpose(0, 2, 3, 1),
                                            order='C')
                    h2e_cas[1] = np.asarray(h2e_cas[1].transpose(0, 2, 3, 1),
                                            order='C')
                    h2e_cas[2] = np.asarray(h2e_cas[2].transpose(0, 2, 3, 1),
                                            order='C')
                    h2e_cas_prime = np.asarray(h2e_cas[1].transpose(2, 0, 1, 3),
                                               order='C')
                else:
                    raise RuntimeError(
                        "WARN: You need to run casscf. Use casscf=True.")

            else:
                mc = mcscf.UCASCI(myhf, norb_cas, nele_cas)
                h1e, ecore = mc.get_h1eff(myhf.mo_coeff)
                h1e_cas = np.array(h1e)
                h2e = mc.get_h2eff(myhf.mo_coeff)
                h2e_cas = np.array(h2e)
                h2e_cas[0] = np.asarray(h2e_cas[0].transpose(0, 2, 3, 1),
                                        order='C')
                h2e_cas[1] = np.asarray(h2e_cas[1].transpose(0, 2, 3, 1),
                                        order='C')
                h2e_cas[2] = np.asarray(h2e_cas[2].transpose(0, 2, 3, 1),
                                        order='C')
                h2e_cas_prime = np.asarray(h2e_cas[1].transpose(2, 0, 1, 3),
                                           order='C')

            # Compute the molecular spin electronic Hamiltonian from the
            # molecular electron integrals
            obi, tbi, core_energy = self.generate_molecular_spin_ham_ur(
                h1e_cas, h2e_cas, h2e_cas_prime, ecore)

        else:

            if nele_cas is None:
                # Compute the 1e integral in atomic orbital then convert to HF basis
                h1e_ao = mol.intor("int1e_kin") + mol.intor("int1e_nuc")
                ## Ways to convert from ao to mo
                #h1e=np.einsum('pi,pq,qj->ij', myhf.mo_coeff, h1e_ao, myhf.mo_coeff)
                h1e = reduce(np.dot, (myhf.mo_coeff.T, h1e_ao, myhf.mo_coeff))
                #h1e=reduce(np.dot, (myhf.mo_coeff.conj().T, h1e_ao, myhf.mo_coeff))

                # Compute the 2e integrals then convert to HF basis
                h2e_ao = mol.intor("int2e_sph", aosym='1')
                h2e = ao2mo.incore.full(h2e_ao, myhf.mo_coeff)

                # Reorder the chemist notation (pq|rs) ERI h_prqs to h_pqrs
                # a_p^dagger a_r a_q^dagger a_s --> a_p^dagger a_q^dagger a_r a_s
                h2e = h2e.transpose(0, 2, 3, 1)

                nuclear_repulsion = myhf.energy_nuc()

                # Compute the molecular spin electronic Hamiltonian from the
                # molecular electron integrals
                obi, tbi, e_nn = self.generate_molecular_spin_ham_restricted(
                    h1e, h2e, nuclear_repulsion)
                energies['nuclear_energy'] = e_nn

            else:

                if integrals_natorb:
                    if spin != 0:
                        raise RuntimeError(
                            "WARN: ROMP2 is unvailable in pyscf.")
                    else:
                        mc = mcscf.CASCI(myhf, norb_cas, nele_cas)
                        h1e_cas, ecore = mc.get_h1eff(natorbs)
                        h2e_cas = mc.get_h2eff(natorbs)
                        h2e_cas = ao2mo.restore('1', h2e_cas, norb_cas)
                        h2e_cas = np.asarray(h2e_cas.transpose(0, 2, 3, 1),
                                             order='C')

                elif integrals_casscf:
                    if casscf:
                        h1e_cas, ecore = mycas.get_h1eff()
                        h2e_cas = mycas.get_h2eff()
                        h2e_cas = ao2mo.restore('1', h2e_cas, norb_cas)
                        h2e_cas = np.asarray(h2e_cas.transpose(0, 2, 3, 1),
                                             order='C')
                    else:
                        raise RuntimeError(
                            "WARN: You need to run casscf. Use casscf=True.")

                else:
                    mc = mcscf.CASCI(myhf, norb_cas, nele_cas)
                    h1e_cas, ecore = mc.get_h1eff(myhf.mo_coeff)
                    h2e_cas = mc.get_h2eff(myhf.mo_coeff)
                    h2e_cas = ao2mo.restore('1', h2e_cas, norb_cas)
                    h2e_cas = np.asarray(h2e_cas.transpose(0, 2, 3, 1),
                                         order='C')

                # Compute the molecular spin electronic Hamiltonian from the
                # molecular electron integrals
                obi, tbi, core_energy = self.generate_molecular_spin_ham_restricted(
                    h1e_cas, h2e_cas, ecore)
                energies['core_energy'] = ecore

        ######################################################
        # Dump energies / etc to a metadata file
        results = {
            'num_electrons': nelec if nele_cas == None else nele_cas,
            'num_orbitals': norb if nele_cas == None else norb_cas,
            'hf_energy': myhf.e_tot,
            'energies': energies,
            'hpq': {
                'data': [(x.real, x.imag)
                         for x in obi.astype(complex).flatten().tolist()]
            },
            'hpqrs': {
                'data': [(x.real, x.imag)
                         for x in tbi.astype(complex).flatten().tolist()]
            },
            'operators': {
                f'{filename}_one_body.dat': 'obi',
                f'{filename}_two_body.dat': 'tbi'
            }
        }

        if cache_data:
            with open(f'{filename}_metadata.json', 'w') as f:
                json.dump(results, f)

        return results

    def generate(self, xyz, basis, **kwargs):
        if xyz == None:
            raise RuntimeError("no molecular geometry provided.")

        if basis == None:
            raise RuntimeError("no basis provided.")

        requiredOptions = ['spin', 'charge']
        for option in requiredOptions:
            if option not in kwargs or kwargs[option] == None:
                raise RuntimeError(
                    f'solvent Hamiltonian generator missing required option - {option}'
                )

        spin = kwargs['spin']
        charge = kwargs['charge']
        symmetry = kwargs['symmetry'] if 'symmetry' in kwargs else False
        memory = kwargs['memory'] if 'memory' in kwargs else 4000
        cycles = kwargs['cycles'] if 'cycles' in kwargs else 100
        initguess = kwargs['initguess'] if 'initguess' in kwargs else 'minao'
        UR = kwargs['UR'] if 'UR' in kwargs else False
        nele_cas = kwargs['nele_cas'] if 'nele_cas' in kwargs else None
        norb_cas = kwargs['norb_cas'] if 'norb_cas' in kwargs else None
        MP2 = kwargs['MP2'] if 'MP2' in kwargs else False
        natorb = kwargs['natorb'] if 'natorb' in kwargs else False
        casci = kwargs['casci'] if 'casci' in kwargs else False
        ccsd = kwargs['ccsd'] if 'ccsd' in kwargs else False
        casscf = kwargs['casscf'] if 'casscf' in kwargs else False
        integrals_natorb = kwargs[
            'integrals_natorb'] if 'integrals_natorb' in kwargs else False
        integrals_casscf = kwargs[
            'integrals_casscf'] if 'integrals_casscf' in kwargs else False
        verbose = kwargs['verbose'] if 'verbose' in kwargs else False
        cache_data = kwargs['cache'] if 'cache' in kwargs else True
        outfilename = kwargs[
            'out_file_name'] if 'out_file_name' in kwargs else None
        return self.get_spin_hamiltonian(xyz, spin, charge, basis, symmetry,
                                         memory, cycles, initguess, UR,
                                         nele_cas, norb_cas, MP2, natorb, casci,
                                         ccsd, casscf, integrals_natorb,
                                         integrals_casscf, verbose, cache_data,
                                         outfilename)


def get_hamiltonian_generator():
    return GasPhaseGenerator()
