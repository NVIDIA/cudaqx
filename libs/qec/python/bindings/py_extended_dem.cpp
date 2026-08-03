/*******************************************************************************
 * Copyright (c) 2026 NVIDIA Corporation & Affiliates.                         *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

// Python bindings for extended_dem, dem_stitch/dem_close operations, and the
// DEM chunk-to-streaming-decoder utility functions.

#include "py_extended_dem.h"
#include "type_casters.h"
#include "cudaq/qec/code_matrices.h"
#include "cudaq/qec/extended_dem.h"

#include <cstdint>
#include <nanobind/nanobind.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/vector.h>

namespace nb = nanobind;

namespace cudaq::qec {

void bindExtendedDem(nb::module_ &mod) {

  // -------------------------------------------------------------------------
  // extended_dem
  // -------------------------------------------------------------------------
  nb::class_<extended_dem>(
      mod, "ExtendedDem",
      "One chunk of a DEM, partitioned into interior, observable, and seam "
      "rows.\n\n"
      "A single extended_dem represents one round (or a pre-stitched group\n"
      "of rounds). Stitch adjacent DEM chunks and close to produce a flat\n"
      "detector_error_model for a decoder.")
      .def(nb::init<>())
      .def("num_faults", &extended_dem::num_faults,
           "Number of fault columns (shared by all four matrices).")
      .def("num_interior", &extended_dem::num_interior,
           "Number of interior detector rows (seams within this DEM chunk).")
      .def("num_observables", &extended_dem::num_observables,
           "Number of observable rows.")
      .def("num_seam_rows", &extended_dem::num_seam_rows,
           "Number of incoming-seam rows (= total checks per round for a\n"
           "uniform per-round chunk).")
      .def("num_in_seam_rows", &extended_dem::num_in_seam_rows,
           "Number of incoming-seam rows. Zero for an init phase chunk.")
      .def("num_out_seam_rows", &extended_dem::num_out_seam_rows,
           "Number of outgoing-seam rows. Zero for a final phase chunk.")
      .def_rw("fault_priors", &extended_dem::fault_priors,
              "Error rates (one per fault column, same order as matrices).")
      .def_rw(
          "in_tags", &extended_dem::in_tags,
          "Check identity tags for the left seam (for dem_stitch validation).")
      .def_rw("out_tags", &extended_dem::out_tags,
              "Check identity tags for the right seam.");

  // -------------------------------------------------------------------------
  // extended_dem_from_css_matrices
  // -------------------------------------------------------------------------
  mod.def("extended_dem_from_css_matrices", &extended_dem_from_css_matrices,
          nb::arg("code"), nb::arg("noise"),
          "Build a one-round ExtendedDem from CSS generator matrices.\n\n"
          "Args:\n"
          "    code:  CssCodes instance.\n"
          "    noise: CssNoise instance.\n"
          "Returns:\n"
          "    ExtendedDem with in_syndrome == out_syndrome (one round).");

  // -------------------------------------------------------------------------
  // dem_chunk_spec / dem_chunks_spec
  // -------------------------------------------------------------------------
  nb::class_<dem_chunk_spec>(
      mod, "DemChunkSpec",
      "One DEM phase, written the way decoder configuration YAML writes it.\n\n"
      "The three H lists are the phase's parity blocks against the same\n"
      "num_faults columns: H_in_sparse is the incoming seam, H_mid_sparse the\n"
      "interior, H_out_sparse the outgoing seam. Each is a flat list of\n"
      "column indices with -1 ending every row, so [0, 1, -1, 2, -1] is two\n"
      "rows. An init phase leaves H_in_sparse empty and a final phase leaves\n"
      "H_out_sparse empty, since nothing precedes the first round or follows\n"
      "the last.")
      .def(nb::init<>())
      .def_rw("num_faults", &dem_chunk_spec::num_faults,
              "Fault columns in this phase; every matrix has this width.")
      .def_rw("H_in_sparse", &dem_chunk_spec::H_in_sparse,
              "Incoming seam rows. Empty for an init phase.")
      .def_rw("H_mid_sparse", &dem_chunk_spec::H_mid_sparse,
              "Interior detector rows owned by this phase.")
      .def_rw("H_out_sparse", &dem_chunk_spec::H_out_sparse,
              "Outgoing seam rows. Empty for a final phase.")
      .def_rw("O_sparse", &dem_chunk_spec::O_sparse,
              "Which fault columns flip each observable (row-sparse, -1 "
              "terminated).")
      .def_rw("error_rates", &dem_chunk_spec::error_rates,
              "One prior per fault column, in column order.")
      .def("is_empty", &dem_chunk_spec::is_empty,
           "True iff no field has been set, meaning the phase is absent.")
      .def("validate", &dem_chunk_spec::validate, nb::arg("context"),
           "Raise ValueError on the first inconsistency, naming `context`.")
      .def("__eq__", [](const dem_chunk_spec &a, const dem_chunk_spec &b) {
        return a == b;
      });

  nb::class_<dem_chunks_spec>(
      mod, "DemChunksSpec",
      "An experiment's DEM as init / bulk / final phases.\n\n"
      "Expanding these for a round count gives the chunk sequence for a whole\n"
      "experiment, so one description serves runs of any length. bulk is\n"
      "optional and only needed when more than two rounds are wanted.")
      .def(nb::init<>())
      .def_rw("init", &dem_chunks_spec::init,
              "First round. Has no incoming seam.")
      .def_rw("bulk", &dem_chunks_spec::bulk,
              "Repeated middle round. Leave empty for a two-round experiment.")
      .def_rw("final", &dem_chunks_spec::final,
              "Last round. Has no outgoing seam.")
      .def("is_empty", &dem_chunks_spec::is_empty,
           "True iff no phase has been set.")
      .def("has_bulk", &dem_chunks_spec::has_bulk,
           "True iff a bulk phase is present and so rounds can be repeated.")
      .def(
          "validate", &dem_chunks_spec::validate,
          "Raise ValueError if the phases cannot stitch into an experiment:\n"
          "a missing init or final, seams that do not line up, priors that do\n"
          "not match the fault count, or indices out of range.")
      .def("__eq__", [](const dem_chunks_spec &a, const dem_chunks_spec &b) {
        return a == b;
      });

  mod.def("dem_chunk_from_spec", &dem_chunk_from_spec, nb::arg("spec"),
          nb::arg("context") = "dem_chunk",
          "Build one ExtendedDem from a DemChunkSpec.\n\n"
          "Args:\n"
          "    spec:    DemChunkSpec to realize.\n"
          "    context: Prefix for any error message.\n"
          "Returns:\n"
          "    ExtendedDem with seam tags assigned so adjacent phases stitch.");

  mod.def(
      "dem_chunks_from_spec", &dem_chunks_from_spec, nb::arg("spec"),
      nb::arg("num_rounds"),
      "Expand phase specs into the chunk sequence for a round count.\n\n"
      "Produces init, num_rounds - 2 copies of bulk, then final -- the list\n"
      "dem_close_all() and dem_stitch_all() consume.\n\n"
      "Args:\n"
      "    spec:       DemChunksSpec describing the phases.\n"
      "    num_rounds: Total rounds counting init and final; at least 2.\n"
      "Returns:\n"
      "    A list of num_rounds ExtendedDem in round order.");

  // -------------------------------------------------------------------------
  // prior_combine_mode
  // -------------------------------------------------------------------------
  nb::enum_<prior_combine_mode>(
      mod, "PriorCombineMode",
      "Strategy for merging fault priors when dem_merge_duplicate_columns()\n"
      "collapses columns with identical row support.")
      .value("or_combine", prior_combine_mode::or_combine,
             "p = 1/2 * (1 - prod(1 - 2 p_i)): exact probability that an odd\n"
             "number of independent events fire (GF(2) / XOR merge). Use for\n"
             "physical fault mechanisms.")
      .value("sum_combine", prior_combine_mode::sum_combine,
             "p = min(1, sum(p_i)): linear approximation valid for small\n"
             "priors.");

  // -------------------------------------------------------------------------
  // dem_merge_duplicate_columns / are_dem_columns_unique /
  // assert_dem_columns_unique / dem_stitch_merged
  // -------------------------------------------------------------------------
  mod.def(
      "dem_merge_duplicate_columns", &dem_merge_duplicate_columns,
      nb::arg("dem"), nb::arg("mode") = prior_combine_mode::or_combine,
      "Merge fault columns with identical row support into single columns.\n\n"
      "Args:\n"
      "    dem:  ExtendedDem whose duplicate columns should be collapsed.\n"
      "    mode: PriorCombineMode controlling how priors are merged.\n"
      "Returns:\n"
      "    New ExtendedDem with unique-support columns.");

  mod.def("are_dem_columns_unique", &are_dem_columns_unique, nb::arg("dem"),
          "Return True iff every fault column has a unique row-support set.");

  mod.def(
      "assert_dem_columns_unique", &assert_dem_columns_unique, nb::arg("dem"),
      "Raise std::invalid_argument if any two columns share row support.\n\n"
      "The error message names the duplicate support and column indices,\n"
      "and suggests calling dem_merge_duplicate_columns() to fix the issue.");

  mod.def(
      "dem_stitch_merged", &dem_stitch_merged, nb::arg("dem_chunks"),
      nb::arg("mode") = prior_combine_mode::or_combine,
      "Stitch DEM chunks left-to-right then merge duplicate columns.\n\n"
      "Equivalent to dem_merge_duplicate_columns(dem_stitch_all(dem_chunks)).\n"
      "Args:\n"
      "    dem_chunks: Non-empty list of ExtendedDem in round order.\n"
      "    mode:       PriorCombineMode for the merge step.\n"
      "Returns:\n"
      "    Stitched ExtendedDem with unique-support columns.");

  // -------------------------------------------------------------------------
  // dem_stitch / dem_stitch_all
  // -------------------------------------------------------------------------
  mod.def("dem_stitch", &dem_stitch, nb::arg("a"), nb::arg("b"),
          "Stitch two adjacent DEM chunks: contract a.out_syndrome with\n"
          "b.in_syndrome. The seam becomes new interior rows in the result.");

  mod.def("dem_stitch_all", &dem_stitch_all, nb::arg("dem_chunks"),
          "Stitch a list of adjacent DEM chunks left-to-right.\n\n"
          "Args:\n"
          "    dem_chunks: Non-empty list of ExtendedDem in round order.\n"
          "Returns:\n"
          "    Fully-stitched ExtendedDem.");

  // -------------------------------------------------------------------------
  // dem_close / dem_close_all
  // -------------------------------------------------------------------------
  mod.def("dem_close", &dem_close, nb::arg("dem"),
          "Collapse an ExtendedDem into a flat detector_error_model.\n\n"
          "Places in_syndrome rows first, then interior rows. out_syndrome is\n"
          "dropped: closing ends the experiment, so put any final-boundary\n"
          "detector in in_syndrome or interior instead.");

  mod.def(
      "dem_close_all", &dem_close_all, nb::arg("dem_chunks"),
      "Build a flat detector_error_model from T DEM chunks in O(T) time.\n\n"
      "Equivalent to dem_close(dem_stitch_all(dem_chunks)) but avoids O(T^2) "
      "cost. The last chunk's out_syndrome is dropped, same as dem_close().\n\n"
      "Args:\n"
      "    dem_chunks: Non-empty list of ExtendedDem in round order.\n"
      "Returns:\n"
      "    detector_error_model ready for any decoder.");

  // -------------------------------------------------------------------------
  // Streaming decoder utilities
  // -------------------------------------------------------------------------
  mod.def("dem_chunk_rounds", &dem_chunk_rounds, nb::arg("dem_chunk"),
          "Return how many measurement rounds a DEM chunk spans.\n\n"
          "One for a DEM chunk from extended_dem_from_css_matrices(); for a\n"
          "stitched DEM chunk, one more than its interior rows divided by its\n"
          "seam rows.");

  mod.def("dem_chunks_to_rounds", &dem_chunks_to_rounds, nb::arg("dem_chunks"),
          "Return the total rounds a list of DEM chunks describes.\n\n"
          "The sum of dem_chunk_rounds() over the list, which equals "
          "len(dem_chunks)\n"
          "only when every DEM chunk spans a single round.");

  mod.def(
      "dem_chunks_to_detector_round", &dem_chunks_to_detector_round,
      nb::arg("dem_chunks"),
      "Return the round index for each detector in "
      "dem_close_all(dem_chunks).\n\n"
      "Returns a list of length T*d where entry i gives the round (0..T-1)\n"
      "of detector i, with T = dem_chunks_to_rounds(dem_chunks). Pass as the\n"
      "'detector_round' parameter to streaming decoders that need to know\n"
      "when each detector becomes available.");

  mod.def(
      "dem_chunks_to_d_sparse", &dem_chunks_to_d_sparse, nb::arg("dem_chunks"),
      "Return the D_sparse measurement-to-detector map from T DEM chunks.\n\n"
      "d_sparse[det_id] lists the raw measurement bit positions that\n"
      "XOR-combine to fire that detector. Compatible with\n"
      "decoder.set_D_sparse().");

  mod.def("dem_chunks_to_o_sparse", &dem_chunks_to_o_sparse,
          nb::arg("dem_chunks"),
          "Return the O_sparse observable-flip map from T DEM chunks.\n\n"
          "o_sparse[obs_id] lists the global fault column indices that flip\n"
          "observable obs_id. Compatible with decoder.set_O_sparse() and\n"
          "to_logical_outcome().");

  mod.def("dem_chunks_to_pcm", &dem_chunks_to_pcm, nb::arg("dem_chunks"),
          "Build a canonicalized CSC parity-check matrix from T DEM chunks.\n\n"
          "Equivalent to closing the chunks, then canonicalizing the detector\n"
          "error matrix and converting it to CSC layout.");
}

} // namespace cudaq::qec
