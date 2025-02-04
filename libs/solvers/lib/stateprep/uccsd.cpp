/****************************************************************-*- C++ -*-****
 * Copyright (c) 2024 NVIDIA Corporation & Affiliates.                         *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/
#include "cudaq/solvers/stateprep/uccsd.h"
#include "cudaq.h"

namespace cudaq::solvers::stateprep {

std::tuple<excitation_list, excitation_list, excitation_list, excitation_list,
           excitation_list>
get_uccsd_excitations(std::size_t numElectrons, std::size_t numQubits,
                      std::size_t spin) {
  if (numQubits % 2 != 0)
    throw std::runtime_error("The total number of qubits should be even.");

  auto numSpatialOrbs = numQubits / 2;
  std::vector<std::size_t> occupiedAlpha, virtualAlpha, occupiedBeta,
      virtualBeta;
  if (spin > 0) {
    auto n_occupied_beta =
        static_cast<std::size_t>(std::floor((float)(numElectrons - spin) / 2));
    auto n_occupied_alpha = numElectrons - n_occupied_beta;
    auto n_virtual_alpha = numSpatialOrbs - n_occupied_alpha;
    auto n_virtual_beta = numSpatialOrbs - n_occupied_beta;

    for (auto i : cudaq::range(n_occupied_alpha))
      occupiedAlpha.push_back(i * 2);

    for (auto i : cudaq::range(n_virtual_alpha))
      virtualAlpha.push_back(i * 2 + numElectrons + 1);

    for (auto i : cudaq::range(n_occupied_beta))
      occupiedBeta.push_back(i * 2 + 1);

    for (auto i : cudaq::range(n_virtual_beta))
      virtualBeta.push_back(i * 2 + numElectrons - 1);
  } else if (numElectrons % 2 == 0 && spin == 0) {
    auto numOccupied =
        static_cast<std::size_t>(std::floor((float)numElectrons / 2));
    auto numVirtual = numSpatialOrbs - numOccupied;

    for (auto i : cudaq::range(numOccupied))
      occupiedAlpha.push_back(i * 2);

    for (auto i : cudaq::range(numVirtual))
      virtualAlpha.push_back(i * 2 + numElectrons);

    for (auto i : cudaq::range(numOccupied))
      occupiedBeta.push_back(i * 2 + 1);

    for (auto i : cudaq::range(numVirtual))
      virtualBeta.push_back(i * 2 + numElectrons + 1);

  } else
    throw std::runtime_error("Incorrect spin multiplicity. Number of electrons "
                             "is odd but spin is 0 " +
                             std::to_string(numElectrons) + ", " +
                             std::to_string(spin));

  excitation_list singlesAlpha, singlesBeta, doublesMixed, doublesAlpha,
      doublesBeta;

  for (auto p : occupiedAlpha)
    for (auto q : virtualAlpha)
      singlesAlpha.push_back({p, q});

  for (auto p : occupiedBeta)
    for (auto q : virtualBeta)
      singlesBeta.push_back({p, q});

  for (auto p : occupiedAlpha)
    for (auto q : occupiedBeta)
      for (auto r : virtualBeta)
        for (auto s : virtualAlpha)
          doublesMixed.push_back({p, q, r, s});

  auto numOccAlpha = occupiedAlpha.size();
  auto numOccBeta = occupiedBeta.size();
  auto numVirtAlpha = virtualAlpha.size();
  auto numVirtBeta = virtualBeta.size();

  for (auto p : cudaq::range(numOccAlpha - 1))
    for (std::size_t q = p + 1; q < numOccAlpha; q++)
      for (auto r : cudaq::range(numVirtAlpha - 1))
        for (std::size_t s = r + 1; s < numVirtAlpha; s++)
          doublesAlpha.push_back({occupiedAlpha[p], occupiedAlpha[q],
                                  virtualAlpha[r], virtualAlpha[s]});

  for (auto p : cudaq::range(numOccBeta - 1))
    for (std::size_t q = p + 1; q < numOccBeta; q++)
      for (auto r : cudaq::range(numVirtBeta - 1))
        for (std::size_t s = r + 1; s < numVirtBeta; s++)
          doublesBeta.push_back({occupiedBeta[p], occupiedBeta[q],
                                 virtualBeta[r], virtualBeta[s]});

  return std::make_tuple(singlesAlpha, singlesBeta, doublesMixed, doublesAlpha,
                         doublesBeta);
}

__qpu__ void single_excitation(cudaq::qview<> qubits, double theta,
                               std::size_t p_occ, std::size_t q_virt) {
  // Y_p X_q
  rx(M_PI_2, qubits[p_occ]);
  h(qubits[q_virt]);

  for (std::size_t i = p_occ; i < q_virt; i++)
    cx(qubits[i], qubits[i + 1]);

  rz(0.5 * theta, qubits[q_virt]);

  for (std::size_t i = q_virt; i > p_occ; i--)
    cx(qubits[i - 1], qubits[i]);

  h(qubits[q_virt]);
  rx(-M_PI_2, qubits[p_occ]);

  // -X_p Y_q
  h(qubits[p_occ]);
  rx(M_PI_2, qubits[q_virt]);

  for (std::size_t i = p_occ; i < q_virt; i++)
    cx(qubits[i], qubits[i + 1]);

  rz(-0.5 * theta, qubits[q_virt]);

  for (std::size_t i = q_virt; i > p_occ; i--)
    cx(qubits[i - 1], qubits[i]);

  rx(-M_PI_2, qubits[q_virt]);
  h(qubits[p_occ]);
}

__qpu__ void double_excitation(cudaq::qview<> qubits, double theta,
                               std::size_t pOcc, std::size_t qOcc,
                               std::size_t rVirt, std::size_t sVirt) {
  std::size_t iOcc = 0, jOcc = 0, aVirt = 0, bVirt = 0;
  if ((pOcc < qOcc) && (rVirt < sVirt)) {
    iOcc = pOcc;
    jOcc = qOcc;
    aVirt = rVirt;
    bVirt = sVirt;
  } else if ((pOcc > qOcc) && (rVirt > sVirt)) {
    iOcc = qOcc;
    jOcc = pOcc;
    aVirt = sVirt;
    bVirt = rVirt;
  } else if ((pOcc < qOcc) && (rVirt > sVirt)) {
    iOcc = pOcc;
    jOcc = qOcc;
    aVirt = sVirt;
    bVirt = rVirt;
    theta *= -1.;
  } else if ((pOcc > qOcc) && (rVirt < sVirt)) {
    iOcc = qOcc;
    jOcc = pOcc;
    aVirt = rVirt;
    bVirt = sVirt;
    theta *= -1.;
  }

  h(qubits[iOcc]);
  h(qubits[jOcc]);
  h(qubits[aVirt]);
  rx(M_PI_2, qubits[bVirt]);

  for (std::size_t i = iOcc; i < jOcc; i++)
    cx(qubits[i], qubits[i + 1]);

  cx(qubits[jOcc], qubits[aVirt]);

  for (std::size_t i = aVirt; i < bVirt; i++)
    cx(qubits[i], qubits[i + 1]);

  rz(0.125 * theta, qubits[bVirt]);

  for (std::size_t i = bVirt; i > aVirt; i--)
    cx(qubits[i - 1], qubits[i]);

  cx(qubits[jOcc], qubits[aVirt]);

  rx(-M_PI_2, qubits[bVirt]);
  h(qubits[aVirt]);

  rx(M_PI_2, qubits[aVirt]);
  h(qubits[bVirt]);

  cx(qubits[jOcc], qubits[aVirt]);
  for (std::size_t i = aVirt; i < bVirt; i++)
    cx(qubits[i], qubits[i + 1]);

  rz(0.125 * theta, qubits[bVirt]);

  for (std::size_t i = bVirt; i > aVirt; i--)
    cx(qubits[i - 1], qubits[i]);
  cx(qubits[jOcc], qubits[aVirt]);

  for (std::size_t i = jOcc; i > iOcc; i--)
    cx(qubits[i - 1], qubits[i]);

  rx(-M_PI_2, qubits[aVirt]);
  h(qubits[jOcc]);

  rx(M_PI_2, qubits[jOcc]);
  h(qubits[aVirt]);

  for (std::size_t i = iOcc; i < jOcc; i++)
    cx(qubits[i], qubits[i + 1]);
  cx(qubits[jOcc], qubits[aVirt]);

  for (std::size_t i = aVirt; i < bVirt; i++)
    cx(qubits[i], qubits[i + 1]);

  rz(-0.125 * theta, qubits[bVirt]);

  for (std::size_t i = bVirt; i > aVirt; i--)
    cx(qubits[i - 1], qubits[i]);
  cx(qubits[jOcc], qubits[aVirt]);

  h(qubits[bVirt]);
  h(qubits[aVirt]);

  rx(M_PI_2, qubits[aVirt]);
  rx(M_PI_2, qubits[bVirt]);

  cx(qubits[jOcc], qubits[aVirt]);
  for (std::size_t i = aVirt; i < bVirt; i++)
    cx(qubits[i], qubits[i + 1]);

  rz(0.125 * theta, qubits[bVirt]);

  for (std::size_t i = bVirt; i > aVirt; i--)
    cx(qubits[i - 1], qubits[i]);

  cx(qubits[jOcc], qubits[aVirt]);

  for (std::size_t i = jOcc; i > iOcc; i--)
    cx(qubits[i - 1], qubits[i]);

  rx(-M_PI_2, qubits[jOcc]);
  h(qubits[iOcc]);

  rx(M_PI_2, qubits[iOcc]);
  h(qubits[jOcc]);

  for (std::size_t i = iOcc; i < jOcc; i++)
    cx(qubits[i], qubits[i + 1]);
  cx(qubits[jOcc], qubits[aVirt]);

  for (std::size_t i = aVirt; i < bVirt; i++)
    cx(qubits[i], qubits[i + 1]);

  rz(0.125 * theta, qubits[bVirt]);

  for (std::size_t i = bVirt; i > aVirt; i--)
    cx(qubits[i - 1], qubits[i]);
  cx(qubits[jOcc], qubits[aVirt]);

  rx(-M_PI_2, qubits[bVirt]);
  rx(-M_PI_2, qubits[aVirt]);

  h(qubits[aVirt]);
  h(qubits[bVirt]);

  cx(qubits[jOcc], qubits[aVirt]);
  for (std::size_t i = aVirt; i < bVirt; i++)
    cx(qubits[i], qubits[i + 1]);

  rz(-0.125 * theta, qubits[bVirt]);

  for (std::size_t i = bVirt; i > aVirt; i--)
    cx(qubits[i - 1], qubits[i]);
  cx(qubits[jOcc], qubits[aVirt]);

  for (std::size_t i = jOcc; i > iOcc; i--)
    cx(qubits[i - 1], qubits[i]);

  h(qubits[bVirt]);
  h(qubits[jOcc]);

  rx(M_PI_2, qubits[jOcc]);
  rx(M_PI_2, qubits[bVirt]);

  for (std::size_t i = iOcc; i < jOcc; i++)
    cx(qubits[i], qubits[i + 1]);

  cx(qubits[jOcc], qubits[aVirt]);

  for (std::size_t i = aVirt; i < bVirt; i++)
    cx(qubits[i], qubits[i + 1]);

  rz(-0.125 * theta, qubits[bVirt]);

  for (std::size_t i = bVirt; i > aVirt; i--)
    cx(qubits[i - 1], qubits[i]);
  cx(qubits[jOcc], qubits[aVirt]);

  rx(-M_PI_2, qubits[bVirt]);
  h(qubits[aVirt]);

  rx(M_PI_2, qubits[aVirt]);
  h(qubits[bVirt]);

  cx(qubits[jOcc], qubits[aVirt]);
  for (std::size_t i = aVirt; i < bVirt; i++)
    cx(qubits[i], qubits[i + 1]);

  rz(-0.125 * theta, qubits[bVirt]);

  for (std::size_t i = bVirt; i > aVirt; i--)
    cx(qubits[i - 1], qubits[i]);
  cx(qubits[jOcc], qubits[aVirt]);

  for (std::size_t i = jOcc; i > iOcc; i--)
    cx(qubits[i - 1], qubits[i]);

  h(qubits[bVirt]);
  rx(-M_PI_2, qubits[aVirt]);
  rx(-M_PI_2, qubits[jOcc]);
  rx(-M_PI_2, qubits[iOcc]);
}

/// @brief Return the number of UCCSD ansatz parameters.
std::size_t get_num_uccsd_parameters(std::size_t numElectrons,
                                     std::size_t numQubits, std::size_t spin) {
  auto [singlesAlpha, singlesBeta, doublesMixed, doublesAlpha, doublesBeta] =
      get_uccsd_excitations(numElectrons, numQubits, spin);
  return singlesAlpha.size() + singlesBeta.size() + doublesMixed.size() +
         doublesAlpha.size() + doublesBeta.size();
}

__qpu__ float positive_floor(float x) {
  int integer_part = (int)x;
  return (float)integer_part;
}

__qpu__ std::size_t getNumOccupiedAlpha(std::size_t numElectrons,
                                        std::size_t spin,
                                        std::size_t numQubits) {
  auto numSpatialOrbs = numQubits / 2;
  if (spin > 0) {
    auto n_occupied_beta = static_cast<std::size_t>(
        positive_floor((float)(numElectrons - spin) / 2));
    auto n_occupied_alpha = numElectrons - n_occupied_beta;
    return n_occupied_alpha;
  }

  auto n_occupied_alpha =
      static_cast<std::size_t>(positive_floor((float)numElectrons / 2));
  return n_occupied_alpha;
}

__qpu__ std::size_t getNumOccupiedBeta(std::size_t numElectrons,
                                       std::size_t spin,
                                       std::size_t numQubits) {

  auto numSpatialOrbs = numQubits / 2;
  if (spin > 0) {
    auto n_occupied_beta = static_cast<std::size_t>(
        positive_floor((float)(numElectrons - spin) / 2));
    return n_occupied_beta;
  }

  auto n_occupied_alpha =
      static_cast<std::size_t>(positive_floor((float)numElectrons / 2));
  return n_occupied_alpha;
}

__qpu__ std::size_t getNumVirtualAlpha(std::size_t numElectrons,
                                       std::size_t spin,
                                       std::size_t numQubits) {

  auto numSpatialOrbs = numQubits / 2;
  if (spin > 0) {
    auto n_occupied_beta = static_cast<std::size_t>(
        positive_floor((float)(numElectrons - spin) / 2));
    auto n_occupied_alpha = numElectrons - n_occupied_beta;
    auto n_virtual_alpha = numSpatialOrbs - n_occupied_alpha;
    return n_virtual_alpha;
  }
  auto n_occupied_alpha =
      static_cast<std::size_t>(positive_floor((float)numElectrons / 2));
  auto n_virtual_alpha = numSpatialOrbs - n_occupied_alpha;
  return n_virtual_alpha;
}

__qpu__ std::size_t getNumVirtualBeta(std::size_t numElectrons,
                                      std::size_t spin, std::size_t numQubits) {

  auto numSpatialOrbs = numQubits / 2;
  if (spin > 0) {
    auto n_occupied_beta = static_cast<std::size_t>(
        positive_floor((float)(numElectrons - spin) / 2));
    auto n_virtual_beta = numSpatialOrbs - n_occupied_beta;
    return n_virtual_beta;
  }

  auto n_occupied_alpha =
      static_cast<std::size_t>(positive_floor((float)numElectrons / 2));
  auto n_virtual_beta = numSpatialOrbs - n_occupied_alpha;
  return n_virtual_beta;
}

__qpu__ void uccsd(cudaq::qview<> qubits, const std::vector<double> &thetas,
                   std::size_t numElectrons, std::size_t spin) {

  int numOccAlpha = getNumOccupiedAlpha(numElectrons, spin, qubits.size());
  int numOccBeta = getNumOccupiedBeta(numElectrons, spin, qubits.size());
  int numVirtAlpha = getNumVirtualAlpha(numElectrons, spin, qubits.size());
  int numVirtBeta = getNumVirtualBeta(numElectrons, spin, qubits.size());
  std::vector<std::size_t> occupiedAlpha(numOccAlpha),
      virtualAlpha(numVirtAlpha), occupiedBeta(numOccBeta),
      virtualBeta(numVirtBeta);
  if (spin > 0) {

    int counter = 0;
    for (std::size_t i = 0; i < numOccAlpha; i++) {
      occupiedAlpha[counter] = i * 2;
      counter++;
    }
    counter = 0;

    for (std::size_t i = 0; i < numVirtAlpha; i++) {
      virtualAlpha[counter] = i * 2 + numElectrons + 1;
      counter++;
    }

    counter = 0;
    for (std::size_t i = 0; i < numOccBeta; i++) {
      occupiedBeta[counter] = i * 2 + 1;
      counter++;
    }

    counter = 0;
    for (std::size_t i = 0; i < numVirtBeta; i++) {
      virtualBeta[counter] = i * 2 + numElectrons - 1;
      counter++;
    }

  } else {
    auto numOccupied = numOccAlpha;
    auto numVirtual = numVirtAlpha;

    int counter = 0;
    for (std::size_t i = 0; i < numOccupied; i++) {
      occupiedAlpha[counter] = i * 2;
      counter++;
    }
    counter = 0;
    for (std::size_t i = 0; i < numVirtual; i++) {
      virtualAlpha[counter] = i * 2 + numElectrons;
      counter++;
    }
    counter = 0;

    for (std::size_t i = 0; i < numOccupied; i++) {
      occupiedBeta[counter] = i * 2 + 1;
      counter++;
    }
    counter = 0;
    for (std::size_t i = 0; i < numVirtual; i++) {
      virtualBeta[counter] = i * 2 + numElectrons + 1;
      counter++;
    }
  }

  std::size_t counter = 0;
  std::vector<std::size_t> singlesAlpha(2 * occupiedAlpha.size() *
                                        virtualAlpha.size());
  for (auto p : occupiedAlpha)
    for (auto q : virtualAlpha) {
      singlesAlpha[counter] = p;
      counter++;
      singlesAlpha[counter] = q;
      counter++;
    }

  counter = 0;
  std::vector<std::size_t> singlesBeta(2 * occupiedBeta.size() *
                                       virtualBeta.size());
  for (auto p : occupiedBeta)
    for (auto q : virtualBeta) {
      singlesBeta[counter] = p;
      counter++;
      singlesBeta[counter] = q;
      counter++;
    }

  counter = 0;
  std::vector<std::size_t> doublesMixed(
      4 * occupiedAlpha.size() * virtualAlpha.size() * occupiedBeta.size() *
      virtualBeta.size());
  for (auto p : occupiedAlpha)
    for (auto q : occupiedBeta)
      for (auto r : virtualBeta)
        for (auto s : virtualAlpha) {
          doublesMixed[counter] = p;
          counter++;
          doublesMixed[counter] = q;
          counter++;
          doublesMixed[counter] = r;
          counter++;
          doublesMixed[counter] = s;
          counter++;
        }

  counter = 0;
  for (int p = 0; p < numOccAlpha - 1; p++)
    for (int q = p + 1; q < numOccAlpha; q++)
      for (int r = 0; r < numVirtAlpha - 1; r++)
        for (int s = r + 1; s < numVirtAlpha; s++)
          counter++;

  std::vector<std::size_t> doublesAlpha(4 * counter);
  counter = 0;
  for (int p = 0; p < numOccAlpha - 1; p++)
    for (int q = p + 1; q < numOccAlpha; q++)
      for (int r = 0; r < numVirtAlpha - 1; r++)
        for (int s = r + 1; s < numVirtAlpha; s++) {
          doublesAlpha[counter] = occupiedAlpha[p];
          counter++;
          doublesAlpha[counter] = occupiedAlpha[q];
          counter++;
          doublesAlpha[counter] = virtualAlpha[r];
          counter++;
          doublesAlpha[counter] = virtualAlpha[s];
          counter++;
        }

  counter = 0;
  for (int p = 0; p < numOccBeta - 1; p++)
    for (int q = p + 1; q < numOccBeta; q++)
      for (int r = 0; r < numVirtBeta - 1; r++)
        for (int s = r + 1; s < numVirtBeta; s++)
          counter++;
  std::vector<std::size_t> doublesBeta(4 * counter);
  counter = 0;
  for (int p = 0; p < numOccBeta - 1; p++)
    for (int q = p + 1; q < numOccBeta; q++)
      for (int r = 0; r < numVirtBeta - 1; r++)
        for (int s = r + 1; s < numVirtBeta; s++) {
          doublesBeta[counter] = occupiedBeta[p];
          counter++;
          doublesBeta[counter] = occupiedBeta[q];
          counter++;
          doublesBeta[counter] = virtualBeta[r];
          counter++;
          doublesBeta[counter] = virtualBeta[s];
          counter++;
        }

  std::size_t thetaCounter = 0;
  for (std::size_t i = 0; i < singlesAlpha.size(); i += 2)
    single_excitation(qubits, thetas[thetaCounter++], singlesAlpha[i],
                      singlesAlpha[i + 1]);

  for (std::size_t i = 0; i < singlesBeta.size(); i += 2)
    single_excitation(qubits, thetas[thetaCounter++], singlesBeta[i],
                      singlesBeta[i + 1]);

  for (std::size_t i = 0; i < doublesMixed.size(); i += 4)
    double_excitation(qubits, thetas[thetaCounter++], doublesMixed[i],
                      doublesMixed[i + 1], doublesMixed[i + 2],
                      doublesMixed[i + 3]);

  for (std::size_t i = 0; i < doublesAlpha.size(); i += 4)
    double_excitation(qubits, thetas[thetaCounter++], doublesAlpha[i],
                      doublesAlpha[i + 1], doublesAlpha[i + 2],
                      doublesAlpha[i + 3]);

  for (std::size_t i = 0; i < doublesBeta.size(); i += 4)
    double_excitation(qubits, thetas[thetaCounter++], doublesBeta[i],
                      doublesBeta[i + 1], doublesBeta[i + 2],
                      doublesBeta[i + 3]);
}

__qpu__ void uccsd(cudaq::qview<> qubits, const std::vector<double> &thetas,
                   std::size_t numElectrons) {
  uccsd(qubits, thetas, numElectrons, 0);
}

} // namespace cudaq::solvers::stateprep