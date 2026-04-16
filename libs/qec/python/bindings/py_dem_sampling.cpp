/*******************************************************************************
 * Copyright (c) 2024 - 2026 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/
#include <algorithm>
#include <cctype>
#include <cmath>
#include <cstdint>
#include <memory>
#include <optional>
#include <random>
#include <string>
#include <vector>

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "type_casters.h"
#include "cudaq/qec/dem_sampling.h"

#include "cuda-qx/core/kwargs_utils.h"

#include <cuda_runtime.h>

namespace py = pybind11;
using namespace pybind11::literals;

namespace cudaq::qec::dem_sampler {

namespace {

/// Convert a py::object to a numpy uint8 array.
py::array_t<uint8_t> asNumpyUint8(py::object obj) {
  py::module_ np = py::module_::import("numpy");
  return np.attr("ascontiguousarray")(obj, "dtype"_a = np.attr("uint8"))
      .cast<py::array_t<uint8_t>>();
}

/// Convert a py::object to a numpy float64 array.
py::array_t<double> asNumpyFloat64(py::object obj) {
  py::module_ np = py::module_::import("numpy");
  auto arr = np.attr("ascontiguousarray")(obj, "dtype"_a = np.attr("float64"))
                 .cast<py::array_t<double>>();
  py::buffer_info info = arr.request();
  if (info.ndim != 1)
    throw std::runtime_error(
        "error_probabilities must be a 1-D array, got ndim=" +
        std::to_string(info.ndim));
  return arr;
}

/// Throw if either input is a torch CPU tensor (no silent numpy conversion).
void rejectTorchCpuTensors(const py::object &check_matrix_obj,
                           const py::object &error_probs_obj) {
  py::module_ torch;
  try {
    torch = py::module_::import("torch");
  } catch (py::error_already_set &) {
    PyErr_Clear();
    return;
  }

  py::object Tensor = torch.attr("Tensor");
  bool check_is_torch = py::isinstance(check_matrix_obj, Tensor);
  bool probs_is_torch = py::isinstance(error_probs_obj, Tensor);

  if (!check_is_torch && !probs_is_torch)
    return;

  auto is_cpu = [&](const py::object &obj) {
    return py::isinstance(obj, Tensor) &&
           !obj.attr("is_cuda").cast<bool>();
  };

  if (is_cpu(check_matrix_obj) || is_cpu(error_probs_obj)) {
    throw std::runtime_error(
        "dem_sampling: PyTorch CPU tensors are not supported. "
        "Convert to NumPy with .numpy() or use CUDA tensors.");
  }
}

enum class DemSamplingBackend { Auto, Cpu, Gpu };

DemSamplingBackend parseBackend(std::string backend) {
  std::transform(
      backend.begin(), backend.end(), backend.begin(),
      [](unsigned char ch) { return static_cast<char>(std::tolower(ch)); });

  if (backend == "auto")
    return DemSamplingBackend::Auto;
  if (backend == "cpu")
    return DemSamplingBackend::Cpu;
  if (backend == "gpu")
    return DemSamplingBackend::Gpu;

  throw std::runtime_error("dem_sampling: invalid backend '" + backend +
                           "'. Expected one of: auto, cpu, gpu.");
}

void validateInputShapes(const py::array_t<uint8_t> &check_matrix_np,
                         const py::array_t<double> &error_probs_np) {
  py::buffer_info h_buf = check_matrix_np.request();
  if (h_buf.ndim != 2)
    throw std::runtime_error("check_matrix must be rank-2");

  py::buffer_info p_buf = error_probs_np.request();
  auto num_mechanisms = static_cast<std::size_t>(h_buf.shape[1]);
  auto num_probs = static_cast<std::size_t>(p_buf.shape[0]);
  if (num_probs != num_mechanisms) {
    throw std::runtime_error("dem_sampling: error_probabilities length (" +
                             std::to_string(num_probs) +
                             ") != check_matrix columns (" +
                             std::to_string(num_mechanisms) + ")");
  }
}

void validateProbabilityValues(const py::array_t<double> &error_probs_np) {
  py::buffer_info p_buf = error_probs_np.request();
  auto *probs = static_cast<const double *>(p_buf.ptr);
  auto count = static_cast<std::size_t>(p_buf.shape[0]);

  for (std::size_t i = 0; i < count; ++i) {
    const double p = probs[i];
    if (!std::isfinite(p) || p < 0.0 || p > 1.0) {
      throw std::runtime_error(
          "dem_sampling: error_probabilities[" + std::to_string(i) +
          "] is invalid. Values must be finite and in [0, 1].");
    }
  }
}

struct CudaDeleter {
  void operator()(void *p) const {
    if (p)
      cudaFree(p);
  }
};

template <typename T>
using DevicePtr = std::unique_ptr<T, CudaDeleter>;

template <typename T>
DevicePtr<T> device_alloc(std::size_t count) {
  T *p = nullptr;
  if (cudaMalloc(&p, count * sizeof(T)) != cudaSuccess) {
    cudaGetLastError();
    return DevicePtr<T>(nullptr);
  }
  return DevicePtr<T>(p);
}

class ScopedCudaDevice {
public:
  ScopedCudaDevice(int target_device, std::string &failure_reason) {
    auto get_device_status = cudaGetDevice(&previous_device);
    if (get_device_status != cudaSuccess) {
      failure_reason = std::string("cudaGetDevice failed: ") +
                       cudaGetErrorString(get_device_status);
      return;
    }

    if (target_device < 0)
      target_device = previous_device;
    active_device = target_device;

    if (previous_device != active_device) {
      auto set_device_status = cudaSetDevice(active_device);
      if (set_device_status != cudaSuccess) {
        failure_reason = std::string("cudaSetDevice failed: ") +
                         cudaGetErrorString(set_device_status);
        return;
      }
    }

    is_active = true;
  }

  ~ScopedCudaDevice() {
    if (is_active && previous_device != active_device)
      cudaSetDevice(previous_device);
  }

  bool ok() const { return is_active; }

private:
  int previous_device = -1;
  int active_device = -1;
  bool is_active = false;
};

bool getTorchDeviceIndex(const py::module_ &torch, const py::object &device,
                         int &device_index, std::string &failure_reason) {
  if (!py::isinstance(device, torch.attr("device"))) {
    failure_reason = "torch device object is invalid";
    return false;
  }

  py::object index_obj = device.attr("index");
  if (!index_obj.is_none()) {
    device_index = index_obj.cast<int>();
  } else {
    auto get_device_status = cudaGetDevice(&device_index);
    if (get_device_status != cudaSuccess) {
      failure_reason = std::string("cudaGetDevice failed: ") +
                       cudaGetErrorString(get_device_status);
      return false;
    }
  }

  int device_count = 0;
  auto count_status = cudaGetDeviceCount(&device_count);
  if (count_status != cudaSuccess) {
    failure_reason = std::string("cudaGetDeviceCount failed: ") +
                     cudaGetErrorString(count_status);
    return false;
  }
  if (device_index < 0 || device_index >= device_count) {
    failure_reason = "invalid CUDA device index " +
                     std::to_string(device_index) +
                     " (device_count=" + std::to_string(device_count) + ")";
    return false;
  }

  return true;
}

bool getTorchCurrentStreamHandle(const py::module_ &torch,
                                 const py::object &device,
                                 std::uintptr_t &stream_handle,
                                 std::string &failure_reason) {
  try {
    py::object stream_obj = torch.attr("cuda").attr("current_stream")(device);
    stream_handle = stream_obj.attr("cuda_stream").cast<std::uintptr_t>();
    return true;
  } catch (py::error_already_set &e) {
    failure_reason =
        std::string("failed to query torch CUDA stream: ") + e.what();
    PyErr_Clear();
    return false;
  }
}

bool tryGpuSampling(const py::array_t<uint8_t> &check_matrix_np,
                    const py::array_t<double> &error_probs_np,
                    std::size_t numShots, unsigned seed,
                    py::array_t<uint8_t> &syndromes_out,
                    py::array_t<uint8_t> &errors_out,
                    std::string &failure_reason) {
  int device_count = 0;
  auto device_count_status = cudaGetDeviceCount(&device_count);
  if (device_count_status != cudaSuccess) {
    failure_reason = std::string("cudaGetDeviceCount failed: ") +
                     cudaGetErrorString(device_count_status);
    return false;
  }
  if (device_count == 0) {
    failure_reason = "no CUDA device available";
    return false;
  }

  py::buffer_info h_buf = check_matrix_np.request();
  if (h_buf.ndim != 2) {
    failure_reason = "check_matrix must be rank-2";
    return false;
  }
  auto num_checks = static_cast<std::size_t>(h_buf.shape[0]);
  auto num_mechanisms = static_cast<std::size_t>(h_buf.shape[1]);

  py::buffer_info p_buf = error_probs_np.request();
  auto num_probs = static_cast<std::size_t>(p_buf.shape[0]);
  if (num_probs != num_mechanisms) {
    failure_reason = "error_probabilities length mismatch";
    return false;
  }

  std::size_t h_bytes = num_checks * num_mechanisms * sizeof(uint8_t);
  std::size_t p_bytes = num_mechanisms * sizeof(double);
  std::size_t syn_bytes = numShots * num_checks * sizeof(uint8_t);
  std::size_t err_bytes = numShots * num_mechanisms * sizeof(uint8_t);

  auto d_H = device_alloc<uint8_t>(num_checks * num_mechanisms);
  if (!d_H) {
    failure_reason = "failed to allocate device memory for check_matrix";
    return false;
  }
  auto d_probs = device_alloc<double>(num_mechanisms);
  if (!d_probs) {
    failure_reason = "failed to allocate device memory for error_probabilities";
    return false;
  }
  auto d_syn = device_alloc<uint8_t>(numShots * num_checks);
  if (!d_syn) {
    failure_reason = "failed to allocate device memory for syndromes";
    return false;
  }
  auto d_err = device_alloc<uint8_t>(numShots * num_mechanisms);
  if (!d_err) {
    failure_reason = "failed to allocate device memory for errors";
    return false;
  }

  auto copy_h_status =
      cudaMemcpy(d_H.get(), h_buf.ptr, h_bytes, cudaMemcpyHostToDevice);
  if (copy_h_status != cudaSuccess) {
    failure_reason = std::string("failed to copy check_matrix to device: ") +
                     cudaGetErrorString(copy_h_status);
    cudaGetLastError();
    return false;
  }
  auto copy_p_status =
      cudaMemcpy(d_probs.get(), p_buf.ptr, p_bytes, cudaMemcpyHostToDevice);
  if (copy_p_status != cudaSuccess) {
    failure_reason = std::string("failed to copy probabilities to device: ") +
                     cudaGetErrorString(copy_p_status);
    cudaGetLastError();
    return false;
  }

  bool ok =
      gpu::sample_dem(d_H.get(), num_checks, num_mechanisms, d_probs.get(),
                      numShots, seed, d_syn.get(), d_err.get());

  if (!ok) {
    failure_reason = "cuStabilizer GPU sampler unavailable at runtime";
    return false;
  }

  syndromes_out =
      py::array_t<uint8_t>({static_cast<py::ssize_t>(numShots),
                            static_cast<py::ssize_t>(num_checks)});
  errors_out =
      py::array_t<uint8_t>({static_cast<py::ssize_t>(numShots),
                            static_cast<py::ssize_t>(num_mechanisms)});

  auto copy_syn_status = cudaMemcpy(syndromes_out.mutable_data(), d_syn.get(),
                                    syn_bytes, cudaMemcpyDeviceToHost);
  if (copy_syn_status != cudaSuccess) {
    failure_reason = std::string("failed to copy syndromes to host: ") +
                     cudaGetErrorString(copy_syn_status);
    cudaGetLastError();
    return false;
  }

  auto copy_err_status = cudaMemcpy(errors_out.mutable_data(), d_err.get(),
                                    err_bytes, cudaMemcpyDeviceToHost);
  if (copy_err_status != cudaSuccess) {
    failure_reason = std::string("failed to copy errors to host: ") +
                     cudaGetErrorString(copy_err_status);
    cudaGetLastError();
    return false;
  }

  failure_reason.clear();
  return true;
}

bool tryTorchGpuSampling(py::object check_matrix_obj,
                         py::object error_probs_obj, std::size_t numShots,
                         unsigned seed, py::tuple &result_out,
                         std::string &failure_reason, bool allow_move_to_cuda) {
  py::module_ torch;
  try {
    torch = py::module_::import("torch");
  } catch (py::error_already_set &) {
    PyErr_Clear();
    py::module_ np = py::module_::import("numpy");
    bool is_numpy = py::isinstance(check_matrix_obj, np.attr("ndarray"));
    if (!is_numpy) {
      PyErr_WarnEx(PyExc_UserWarning,
                   "[cudaq_qec.dem_sampling] PyTorch is not installed. "
                   "Install it with: pip install torch",
                   1);
    }
    failure_reason = "PyTorch is not available";
    return false;
  }

  if (!py::isinstance(check_matrix_obj, torch.attr("Tensor")) ||
      !py::isinstance(error_probs_obj, torch.attr("Tensor"))) {
    failure_reason = "inputs are not both torch.Tensor";
    return false;
  }

  py::object check_t = check_matrix_obj;
  py::object probs_t = error_probs_obj;

  bool check_is_cuda = check_t.attr("is_cuda").cast<bool>();
  bool probs_is_cuda = probs_t.attr("is_cuda").cast<bool>();

  py::object device;
  if (check_is_cuda) {
    device = check_t.attr("device");
  } else if (probs_is_cuda) {
    device = probs_t.attr("device");
  } else if (allow_move_to_cuda) {
    int device_count = 0;
    auto device_count_status = cudaGetDeviceCount(&device_count);
    if (device_count_status != cudaSuccess) {
      failure_reason = std::string("cudaGetDeviceCount failed: ") +
                       cudaGetErrorString(device_count_status);
      return false;
    }
    if (device_count == 0) {
      failure_reason = "no CUDA device available";
      return false;
    }
    device = torch.attr("device")("cuda");
  } else {
    failure_reason = "torch tensors are not on CUDA device";
    return false;
  }

  check_t =
      check_t.attr("to")("device"_a = device, "dtype"_a = torch.attr("uint8"));
  probs_t = probs_t.attr("to")("device"_a = device,
                               "dtype"_a = torch.attr("float64"));
  check_t = check_t.attr("contiguous")();
  probs_t = probs_t.attr("contiguous")();

  auto check_shape = check_t.attr("shape").cast<py::tuple>();
  if (check_shape.size() != 2) {
    failure_reason = "check_matrix must be rank-2";
    return false;
  }

  auto probs_shape = probs_t.attr("shape").cast<py::tuple>();
  if (probs_shape.size() != 1) {
    failure_reason = "error_probabilities must be rank-1";
    return false;
  }

  auto num_checks =
      static_cast<std::size_t>(check_shape[0].cast<py::ssize_t>());
  auto num_mechanisms =
      static_cast<std::size_t>(check_shape[1].cast<py::ssize_t>());
  auto num_probs = static_cast<std::size_t>(probs_shape[0].cast<py::ssize_t>());

  if (num_probs != num_mechanisms) {
    failure_reason = "error_probabilities length mismatch";
    return false;
  }

  bool all_finite =
      torch.attr("isfinite")(probs_t).attr("all")().attr("item")().cast<bool>();
  bool any_below_zero =
      probs_t.attr("lt")(0.0).attr("any")().attr("item")().cast<bool>();
  bool any_above_one =
      probs_t.attr("gt")(1.0).attr("any")().attr("item")().cast<bool>();
  if (!all_finite || any_below_zero || any_above_one) {
    failure_reason = "error_probabilities must be finite values in [0, 1]";
    return false;
  }

  py::object actual_device = check_t.attr("device");
  int target_device = -1;
  if (!getTorchDeviceIndex(torch, actual_device, target_device, failure_reason))
    return false;

  ScopedCudaDevice device_guard(target_device, failure_reason);
  if (!device_guard.ok())
    return false;

  std::uintptr_t stream_handle = 0;
  if (!getTorchCurrentStreamHandle(torch, actual_device, stream_handle,
                                   failure_reason))
    return false;

  py::object syndromes_t =
      torch.attr("empty")(py::make_tuple(static_cast<py::ssize_t>(numShots),
                                         static_cast<py::ssize_t>(num_checks)),
                          "dtype"_a = torch.attr("uint8"), "device"_a = device);
  py::object errors_t = torch.attr("empty")(
      py::make_tuple(static_cast<py::ssize_t>(numShots),
                     static_cast<py::ssize_t>(num_mechanisms)),
      "dtype"_a = torch.attr("uint8"), "device"_a = device);

  auto d_H = reinterpret_cast<const uint8_t *>(
      check_t.attr("data_ptr")().cast<std::uintptr_t>());
  auto d_probs = reinterpret_cast<const double *>(
      probs_t.attr("data_ptr")().cast<std::uintptr_t>());
  auto d_syn = reinterpret_cast<uint8_t *>(
      syndromes_t.attr("data_ptr")().cast<std::uintptr_t>());
  auto d_err = reinterpret_cast<uint8_t *>(
      errors_t.attr("data_ptr")().cast<std::uintptr_t>());

  bool ok = gpu::sample_dem(d_H, num_checks, num_mechanisms, d_probs, numShots,
                            seed, d_syn, d_err, stream_handle);
  if (!ok) {
    failure_reason = "cuStabilizer GPU sampler unavailable at runtime";
    return false;
  }

  result_out = py::make_tuple(syndromes_t, errors_t);
  failure_reason.clear();
  return true;
}

} // namespace

void bindDemSampling(py::module &mod) {
  auto qecmod = py::hasattr(mod, "qecrt")
                    ? mod.attr("qecrt").cast<py::module_>()
                    : mod.def_submodule("qecrt");

  qecmod.def(
      "dem_sampling",
      [](py::object check_matrix_obj, std::size_t numShots,
         py::object error_probs_obj, std::optional<unsigned> seed,
         std::string backend) -> py::tuple {
        const auto backend_mode = parseBackend(std::move(backend));

        if (numShots == 0) {
          auto check_matrix_np = asNumpyUint8(check_matrix_obj);
          auto error_probs_np = asNumpyFloat64(error_probs_obj);
          validateInputShapes(check_matrix_np, error_probs_np);
          py::buffer_info h_buf = check_matrix_np.request();
          auto num_checks = static_cast<py::ssize_t>(h_buf.shape[0]);
          auto num_mechanisms = static_cast<py::ssize_t>(h_buf.shape[1]);
          py::module_ np = py::module_::import("numpy");
          return py::make_tuple(
              np.attr("empty")(py::make_tuple(0, num_checks),
                               "dtype"_a = np.attr("uint8")),
              np.attr("empty")(py::make_tuple(0, num_mechanisms),
                               "dtype"_a = np.attr("uint8")));
        }

        unsigned actual_seed =
            seed.has_value() ? seed.value()
                             : static_cast<unsigned>(std::random_device{}());

        std::string torch_gpu_failure_reason = "not attempted";
        if (backend_mode != DemSamplingBackend::Cpu) {
          py::tuple torch_result;
          bool allow_move_to_cuda = backend_mode == DemSamplingBackend::Gpu;
          if (tryTorchGpuSampling(check_matrix_obj, error_probs_obj, numShots,
                                  actual_seed, torch_result,
                                  torch_gpu_failure_reason, allow_move_to_cuda))
            return torch_result;
        }

        rejectTorchCpuTensors(check_matrix_obj, error_probs_obj);

        auto check_matrix_np = asNumpyUint8(check_matrix_obj);
        auto error_probs_np = asNumpyFloat64(error_probs_obj);
        validateInputShapes(check_matrix_np, error_probs_np);
        validateProbabilityValues(error_probs_np);

        if (backend_mode != DemSamplingBackend::Cpu) {
          py::array_t<uint8_t> syndromes_gpu, errors_gpu;
          std::string gpu_failure_reason;
          if (tryGpuSampling(check_matrix_np, error_probs_np, numShots,
                             actual_seed, syndromes_gpu, errors_gpu,
                             gpu_failure_reason))
            return py::make_tuple(syndromes_gpu, errors_gpu);

          if (backend_mode == DemSamplingBackend::Gpu) {
            throw std::runtime_error(
                "dem_sampling: GPU backend requested but unavailable. "
                "torch path: " +
                torch_gpu_failure_reason +
                "; numpy path: " + gpu_failure_reason);
          }
        }

        auto H = cudaqx::toTensor(check_matrix_np);

        py::buffer_info p_buf = error_probs_np.request();
        std::vector<double> probs(static_cast<double *>(p_buf.ptr),
                                  static_cast<double *>(p_buf.ptr) +
                                      p_buf.shape[0]);

        auto [syndromes, errors] =
            cpu::sample_dem(H, numShots, probs, actual_seed);

        return py::make_tuple(
            cudaq::python::copyCUDAQXTensorToPyArray(syndromes),
            cudaq::python::copyCUDAQXTensorToPyArray(errors));
      },
      R"pbdoc(
        Sample syndrome measurements from a detector error model.

        Generates random error vectors according to the given per-mechanism
        error probabilities, then computes syndromes via the check matrix.
        Backend selection is controlled by the `backend` argument.
          - "auto" (default): try GPU first, fall back to CPU.
          - "cpu": always run the CPU implementation.
          - "gpu": require GPU; raise if unavailable.

        The check_matrix and error_probabilities arguments accept NumPy
        arrays or PyTorch CUDA tensors (requires user-installed torch).

        When PyTorch CUDA tensors are provided on the GPU path, outputs are
        returned as PyTorch CUDA tensors. Otherwise, outputs are NumPy arrays.

        PyTorch CPU tensors are not supported. Convert to NumPy first.

        Args:
            check_matrix: Binary check matrix [num_checks x num_error_mechanisms],
                          as a NumPy uint8 array or PyTorch CUDA tensor.
            numShots: Number of measurement shots to sample.
            error_probabilities: Per-error-mechanism probabilities
                                 [num_error_mechanisms], as a NumPy float64
                                 array or PyTorch CUDA tensor.
            seed: Optional RNG seed for reproducibility.
            backend: "auto", "cpu", or "gpu". Default is "auto".

        Returns:
            A tuple (syndromes, errors) with shapes [numShots x num_checks]
            and [numShots x num_error_mechanisms].
      )pbdoc",
      py::arg("check_matrix"), py::arg("numShots"),
      py::arg("error_probabilities"), py::arg("seed") = py::none(),
      py::arg("backend") = "auto");
}

} // namespace cudaq::qec::dem_sampler
