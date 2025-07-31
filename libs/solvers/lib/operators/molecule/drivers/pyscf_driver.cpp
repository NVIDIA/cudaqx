/*******************************************************************************
 * Copyright (c) 2022 - 2025 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#include "nlohmann/json.hpp"

#include "cuda-qx/core/library_utils.h"
#include "cuda-qx/core/tensor.h"
#include "process.h"
#include "cudaq/solvers/operators/molecule/fermion_compiler.h"
#include "cudaq/solvers/operators/molecule/molecule_package_driver.h"

#include "common/Logger.h"
#include "common/RestClient.h"

#include <filesystem>
#include <fmt/core.h>
#include <thread>

using namespace cudaqx;

namespace cudaq::solvers {

// Create a tear down service
class PySCFTearDown : public tear_down {
private:
  pid_t pid;

public:
  PySCFTearDown(pid_t p) : pid(p) {}
  void runTearDown() const {
    // shut down the web server
    [[maybe_unused]] auto success = ::kill(pid, SIGTERM);
    using namespace std::chrono_literals;
    std::this_thread::sleep_for(100ms);
  }
};

class RESTPySCFDriver : public MoleculePackageDriver {

public:
  CUDAQ_ADD_EXTENSION_CREATOR_FUNCTION(MoleculePackageDriver, RESTPySCFDriver)

  bool is_available() const override {
    cudaq::RestClient client;
    std::map<std::string, std::string> headers;
    try {
      auto res = client.get("localhost:8000/", "status", headers);
      if (res.contains("status") &&
          res["status"].get<std::string>() == "available")
        return true;
    } catch (std::exception &e) {
      return false;
    }
    return true;
  }

  std::unique_ptr<tear_down>
  make_available(const std::string &python_path) const override {

    // Start up the web service, if failed, return nullptr
    std::filesystem::path libPath{cudaqx::__internal__::getCUDAQXLibraryPath(
        cudaqx::__internal__::CUDAQXLibraryType::Solvers)};
    auto cudaqLibPath = libPath.parent_path();
    auto cudaqPySCFTool = cudaqLibPath.parent_path() / "bin" / "cudaq-pyscf";
    auto argString = cudaqPySCFTool.string() + " --server-mode";
    if (!python_path.empty())
      argString = python_path + " " + argString;
    int a0, a1;
    auto [ret, msg] = cudaqx::launchProcess(argString.c_str());
    if (ret == -1)
      return nullptr;

    if (!msg.empty()) {
      cudaq::info("pyscf error: {}", libPath.parent_path().string());
      cudaq::info("pyscf error: {}", msg);
      cudaq::info("advice - check `lsof -n -i :8000` for dead pyscf process. "
                  "kill it.");
      throw std::runtime_error(
          "error encountered when launching pyscf molecule generation server.");
    }

    cudaq::RestClient client;
    using namespace std::chrono_literals;
    std::size_t ticker = 0;
    std::map<std::string, std::string> headers{
        {"Content-Type", "application/json"}};
    while (true) {
      std::this_thread::sleep_for(100ms);

      nlohmann::json metadata;
      try {
        metadata = client.get("localhost:8000/", "status", headers);
        if (metadata.count("status"))
          break;
      } catch (...) {
        continue;
      }

      if (ticker > 5000)
        return nullptr;

      ticker += 100;
    }

    return std::make_unique<PySCFTearDown>(ret);
  }

  /// @brief Create the molecular hamiltonian
  molecular_hamiltonian createMolecule(const molecular_geometry &geometry,
                                       const std::string &basis, int spin,
                                       int charge,
                                       molecule_options options) override {
    std::string xyzFileStr = "";
    // Convert the geometry to an XYZ string
    for (auto &atom : geometry)
      xyzFileStr +=
          fmt::format("{} {:f} {:f} {:f}; ", atom.name, atom.coordinates[0],
                      atom.coordinates[1], atom.coordinates[2]);

    cudaq::RestClient client;
    nlohmann::json payload = {{"xyz", xyzFileStr},
                              {"basis", basis},
                              {"spin", spin},
                              {"charge", charge},
                              {"type", "gas_phase"},
                              {"symmetry", false},
                              {"cycles", options.cycles},
                              {"initguess", options.initguess},
                              {"UR", options.UR},
                              {"MP2", options.MP2},
                              {"natorb", options.natorb},
                              {"casci", options.casci},
                              {"ccsd", options.ccsd},
                              {"casscf", options.casscf},
                              {"integrals_natorb", options.integrals_natorb},
                              {"integrals_casscf", options.integrals_casscf},
                              {"verbose", options.verbose}};
    if (options.nele_cas.has_value())
      payload["nele_cas"] = options.nele_cas.value();
    if (options.norb_cas.has_value())
      payload["norb_cas"] = options.norb_cas.value();
    if (options.potfile.has_value())
      payload["potfile"] = options.potfile.value();

    std::map<std::string, std::string> headers{
        {"Content-Type", "application/json"}};
    auto metadata = client.post("localhost:8000/", "create_molecule", payload,
                                headers, true);

    // Get the energy, num orbitals, and num qubits
    std::unordered_map<std::string, double> energies;
    for (auto &[energyName, E] : metadata["energies"].items())
      energies.insert({energyName, E});

    double energy = 0.0;
    if (energies.contains("nuclear_energy"))
      energy = energies["nuclear_energy"];
    else if (energies.contains("core_energy"))
      energy = energies["core_energy"];

    auto numOrb = metadata["num_orbitals"].get<std::size_t>();
    auto numQubits = 2 * numOrb;
    auto num_electrons = metadata["num_electrons"].get<std::size_t>();

    // Get the operators
    auto hpqElements = metadata["hpq"]["data"];
    auto hpqrsElements = metadata["hpqrs"]["data"];
    std::vector<std::complex<double>> hpqValues, hpqrsValues;
    for (auto &element : hpqElements)
      hpqValues.push_back({element[0].get<double>(), element[1].get<double>()});
    for (auto &element : hpqrsElements)
      hpqrsValues.push_back(
          {element[0].get<double>(), element[1].get<double>()});

    cudaqx::tensor hpq, hpqrs;
    hpq.copy(hpqValues.data(), {numQubits, numQubits});
    hpqrs.copy(hpqrsValues.data(),
               {numQubits, numQubits, numQubits, numQubits});

    // Transform to a spin operator
    auto transform = fermion_compiler::get(options.fermion_to_spin);
    auto spinHamiltonian = transform->generate(energy, hpq, hpqrs);

    // Return the molecular hamiltonian
    return molecular_hamiltonian{spinHamiltonian, hpq,    hpqrs,
                                 num_electrons,   numOrb, energies};
  }
};
CUDAQ_REGISTER_EXTENSION(RESTPySCFDriver)

} // namespace cudaq::solvers
