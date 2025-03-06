#include "test_kernels.h"

#include "cudaq/solvers/stateprep/uccsd.h"

__qpu__ void hartreeFock2Electrons(cudaq::qvector<> &q) {
  for (std::size_t i = 0; i < 2; i++)
    x(q[i]);
}

__qpu__ void statePrepNElectrons(cudaq::qvector<> &q,
                                 std::size_t numElectrons) {
  for (std::size_t i = 0; i < numElectrons; i++)
    x(q[i]);
}

__qpu__ void statePrep4Electrons(cudaq::qvector<> &q) {
  for (std::size_t i = 0; i < 4; i++)
    x(q[i]);
}

__qpu__ void statePrep6Electrons(cudaq::qvector<> &q) {
  for (std::size_t i = 0; i < 6; i++)
    x(q[i]);
}

__qpu__ void ansatz(std::vector<double> theta) {
  cudaq::qvector q(2);
  x(q[0]);
  ry(theta[0], q[1]);
  x<cudaq::ctrl>(q[1], q[0]);
}

__qpu__ void ansatzNonStdSignature(double theta, int N) {
  cudaq::qvector q(N);
  x(q[0]);
  ry(theta, q[1]);
  x<cudaq::ctrl>(q[1], q[0]);
}

__qpu__ void callUccsdStatePrep(std::vector<double> params) {
  cudaq::qvector q(4);
  for (auto i : cudaq::range(2))
    x(q[i]);

  cudaq::solvers::stateprep::uccsd(q, params, 2, 0);
}

__qpu__ void callUccsdStatePrepWithArgs(std::vector<double> params,
                                        std::size_t numQubits,
                                        std::size_t numElectrons) {
  cudaq::qvector q(numQubits);
  for (auto i : cudaq::range(numElectrons))
    x(q[i]);

  cudaq::solvers::stateprep::uccsd(q, params, numElectrons);
}
