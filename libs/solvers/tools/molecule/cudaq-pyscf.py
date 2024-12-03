#!/usr/bin/env python3

import argparse
import importlib, pkgutil
import cudaq_solvers.tools.molecule.pyscf.generators

from fastapi import FastAPI, Response
from pydantic import BaseModel, PlainValidator, PlainSerializer
import uvicorn, os, signal, importlib, pkgutil
from typing import List, Annotated
import numpy as np


def iter_namespace(ns_pkg):
    return pkgutil.iter_modules(ns_pkg.__path__, ns_pkg.__name__ + ".")


discovered_plugins = {}
for finder, name, ispkg in iter_namespace(
        cudaq_solvers.tools.molecule.pyscf.generators):
    try:
        discovered_plugins[name] = importlib.import_module(name)
    except ModuleNotFoundError:
        pass

hamiltonianGenerators = {
    plugin.get_hamiltonian_generator().name(): plugin
    for _, plugin in discovered_plugins.items()
}

#############################
# Argument Parser
#############################

parser = argparse.ArgumentParser()

parser.add_argument('--server-mode', action='store_true', default=False)

# Add arguments
parser.add_argument(
    '--type',
    type=str,
    help='type of simulation (hamiltonian generator) - options include {}'.
    format([k for k, v in hamiltonianGenerators.items()]),
    default='gas_phase')
parser.add_argument('--xyz', help="xyz file", type=str)
parser.add_argument('--basis', help='', type=str)
parser.add_argument('--charge', help="charge of the system", type=int)
parser.add_argument('--out-file-name',
                    help='base file name for output data.',
                    type=str)
parser.add_argument('--spin', help="no. of unpaired electrons (2 *s)", type=int)
parser.add_argument('--symmetry', help="", action='store_true', default=False)
parser.add_argument('--memory', help="", type=float, default=4000)
parser.add_argument('--cycles', help="", type=int, default=100)
parser.add_argument('--initguess', help="", type=str, default='minao')
parser.add_argument('--UR', help="", action='store_true', default=False)
parser.add_argument('--MP2', help="", action='store_true', default=False)
parser.add_argument('--nele_cas', help="", type=int, default=None)
parser.add_argument('--norb_cas', help="", type=int, default=None)
parser.add_argument('--natorb', help="", action='store_true', default=False)
parser.add_argument('--casci', help="", action='store_true', default=False)
parser.add_argument('--ccsd', help="", action='store_true', default=False)
parser.add_argument('--casscf', help="", action='store_true', default=False)
parser.add_argument('--integrals_natorb',
                    help="",
                    action='store_true',
                    default=False)
parser.add_argument('--integrals_casscf',
                    help="",
                    action='store_true',
                    default=False)
parser.add_argument('--potfile', help="", type=str, default=None)
parser.add_argument('--verbose',
                    help="Verbose printout",
                    action='store_true',
                    default=False)

# Parse the arguments
args = parser.parse_args()

if not args.server_mode:

    if args.type not in hamiltonianGenerators:
        raise RuntimeError(f'invalid hamiltonian generator type - {args.type}')
    hamiltonianGenerator = hamiltonianGenerators[
        args.type].get_hamiltonian_generator()

    filterArgs = ['xyz', 'basis']
    filteredArgs = {
        k: v for (k, v) in vars(args).items() if k not in filterArgs
    }
    res = hamiltonianGenerator.generate(args.xyz, args.basis, **filteredArgs)
    print(res)

    exit(0)

app = FastAPI()


@app.get("/shutdown")
async def shutdown():
    os.kill(os.getpid(), signal.SIGTERM)
    return Response(status_code=200, content='Server shutting down...')


class IntegralsData(BaseModel):
    data: List[List]


class MoleculeInput(BaseModel):
    basis: str
    xyz: str
    spin: int
    charge: int
    type: str = 'gas_phase'
    symmetry: bool = False
    cycles: int = 100
    memory: float = 4000.
    initguess: str = 'minao'
    UR: bool = False
    MP2: bool = False
    natorb: bool = False
    casci: bool = False
    ccsd: bool = False
    casscf: bool = False
    integrals_natorb: bool = False
    integrals_casscf: bool = False
    verbose: bool = False
    nele_cas: int = None
    norb_cas: int = None
    potfile: str = None


class Molecule(BaseModel):
    energies: dict
    num_orbitals: int
    num_electrons: int
    hpq: IntegralsData
    hpqrs: IntegralsData


@app.get("/status")
async def get_status():
    return {"status": "available"}


@app.post("/create_molecule")
async def create_molecule(molecule: MoleculeInput):
    hamiltonianGenerator = hamiltonianGenerators[
        molecule.type].get_hamiltonian_generator()

    filterArgs = ['xyz', 'basis']
    filteredArgs = {
        k: v for (k, v) in vars(molecule).items() if k not in filterArgs
    }
    filteredArgs['cache_data'] = False
    res = hamiltonianGenerator.generate(molecule.xyz, molecule.basis,
                                        **filteredArgs)
    return Molecule(energies=res['energies'],
                    num_orbitals=res['num_orbitals'],
                    num_electrons=res['num_electrons'],
                    hpq=IntegralsData(data=res['hpq']['data']),
                    hpqrs=IntegralsData(data=res['hpqrs']['data']))


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level='critical')
