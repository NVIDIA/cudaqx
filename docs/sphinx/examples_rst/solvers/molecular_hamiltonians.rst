Generating Molecular Hamiltonians
----------------------------------

The CUDA-Q Solvers library accelerates a wide range of applications in the domain of quantum chemistry.
To facilitate these calculations, CUDA-Q Solvers provides the `solver.create_molecule` function to allow users to generate basis sets and Hamiltonians for many systems of interest.
The molecule class contains basis set informations, and the Hamiltonian (`molecule.hamiltonian`) for the target systems.
These Hamiltonians can then be used as input into the hybrid quantum-classical solvers that the CUDA-Q Solvers API provides.


Molecular Orbitals and Hamiltonians
+++++++++++++++++++++++++++++++++++

First we define the atomic geometry of the molecule by specifying a array of atomic symbols as strings, and coordinates in 3D space. We then get a molecule object from the `solvers.create_molecule` call.
Here we create "default" Hamiltonian for the N2 system using complete active space molecular orbitals constructed from Restricted Hartree-Fock molecular orbitals.

.. tab:: Python

  .. code-block:: python

    geometry = [('N', (0.0, 0.0, 0.5600)), ('N', (0.0, 0.0, -0.5600))]
    molecule = solvers.create_molecule(geometry,
                                       'sto-3g',
                                       0,
                                       0,
                                       nele_cas=2,
                                       norb_cas=3,
                                       verbose=True)

We specify:
  - The geometry previously created
  - The single particle basis set (here STO-3G)
  - The total spin
  - The total charge
  - The number of electrons in the complete active space
  - The number of orbitals in the complete activate space
  - A verbosity flag to help introspect on the data what was generated.

Along with the orbitals and Hamiltonian, we can also view various properties like the Hartree-Fock energy, and the energy of the frozen core orbitals by printing `molecule.energies`.

For using Unrestricted Hartree-Fock (UHF) orbitals, you can set the `UR` parameter to `True` in the `create_molecule` function. Here's an example:

.. tab:: Python

  .. code-block:: python

    geometry = [('N', (0.0, 0.0, 0.5600)), ('N', (0.0, 0.0, -0.5600))]
    molecule = solvers.create_molecule(geometry,
                                       'sto-3g',
                                       0,
                                       0,
                                       nele_cas=2,
                                       norb_cas=3,
                                       UR=True,
                                       verbose=True)


Natural Orbitals from MP2
++++++++++++++++++++++++++
Now we take our same N2 molecule, but generate natural orbitals from second order Møller–Plesset perturbation theory as the basis.

.. tab:: Python

  .. code-block:: python

    geometry = [('N', (0.0, 0.0, 0.5600)), ('N', (0.0, 0.0, -0.5600))]
    molecule = solvers.create_molecule(geometry,
                                       'sto-3g',
                                       0,
                                       0,
                                       nele_cas=2,
                                       norb_cas=3,
                                       MP2=True,
                                       integrals_natorb=True,
                                       verbose=True)

Note that we use the same API but,toggle `MP2=True` and `integrals_natorb=True`. This will generate the molecular orbitals from MP2 natural orbitals, and compute the Hamiltonian integrals in this basis.
This option is unallowed yet when using `UR=True`. When using `UR=True`, only UHF molecular orbitals are employed to generate the electronic Hamiltonian.

CASSCF Orbitals
+++++++++++++++

Next, we can start from either Hartree-Fock or perturbation theory atomic orbitals and build complete active space self-consistent field (CASSCF) molecular orbitals.

.. tab:: Python

  .. code-block:: python

    geometry = [('N', (0.0, 0.0, 0.5600)), ('N', (0.0, 0.0, -0.5600))]
    molecule = solvers.create_molecule(geometry,
                                       'sto-3g',
                                       0,
                                       0,
                                       nele_cas=2,
                                       norb_cas=3,
                                       casscf=True,
                                       integrals_casscf=True,
                                       verbose=True)

For Hartree-Fock, or

.. tab:: Python

  .. code-block:: python

    geometry = [('N', (0.0, 0.0, 0.5600)), ('N', (0.0, 0.0, -0.5600))]
    molecule = solvers.create_molecule(geometry,
                                       'sto-3g',
                                       0,
                                       0,
                                       nele_cas=2,
                                       norb_cas=3,
                                       MP2=True,
                                       natorb=True,
                                       casscf=True,
                                       integrals_casscf=True,
                                       verbose=True)

for MP2 natural orbitals. In these cases, printing the `molecule.energies` also shows the `R-CASSCF` energy for the system.

Now that we have seen how to generate basis sets and Hamiltonians for quantum chemistry systems, we can use these as inputs to hybrid quantum-classical methods like VQE or adapt VQE via the CUDA-Q Solvers API.


For open-shell systems
++++++++++++++++++++++++++

If you want to use Restricted Open-shell Hartree-Fock (ROHF) orbitals, you can set the `spin` parameter to a non-zero value while keeping the `charge` parameter as needed. For example, for a molecule with one unpaired electron, you can set `spin=1` and `charge=1`. Here's an example:

.. tab:: Python

  .. code-block:: python

    geometry = [('N', (0.0, 0.0, 0.5600)), ('N', (0.0, 0.0, -0.5600))]
    molecule = solvers.create_molecule(geometry,
                                       'sto-3g',
                                       1,
                                       1,
                                       nele_cas=3,
                                       norb_cas=3,
                                       verbose=True)


If you want to use Unrestricted Hartree-Fock (UHF) orbitals, you can set `UR=True` and the `spin` parameter to a non-zero value while keeping the `charge` parameter as needed. Here's an example:

.. tab:: Python

  .. code-block:: python

    geometry = [('N', (0.0, 0.0, 0.5600)), ('N', (0.0, 0.0, -0.5600))]
    molecule = solvers.create_molecule(geometry,
                                       'sto-3g',
                                       1,
                                       1,
                                       nele_cas=3,
                                       norb_cas=3,
                                       UR=True,
                                       verbose=True)