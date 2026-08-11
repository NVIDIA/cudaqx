# SoftRoCE (rxe) on MLNX/DOCA-OFED hosts

Two independent gaps stop `--roce-pair rxe` from working out of the box on
hosts that run DOCA-OFED (validated on a GB200, `6.17.0-1008-nvidia-64k` +
DOCA-OFED 25.10):

1. **Kernel:** the OFED DKMS stack replaces `ib_core` (in
   `/lib/modules/*/updates/dkms/`), whose exported symbol CRCs differ from
   the in-tree build, so the distro's `rdma_rxe.ko` fails to load with
   `disagrees about version of symbol ib_*` (err -22) — and MLNX OFED
   dropped the rxe driver from its own source tree (only the uapi header
   remains in `ofa_kernel`).
2. **Userspace:** the Mellanox rdma-core fork's `ibverbs-providers` ships
   only the mlx5 provider. Its provider ABI (`rdmav59`) differs from
   Ubuntu's rdma-core, so the Ubuntu package can't supply `librxe` either;
   in their source release the rxe provider exists but is `if (0)`-disabled
   in `CMakeLists.txt`.

## What this directory provides (gap 1)

- `ofed-compat.patch` — three small deltas that make the **upstream v6.17**
  `drivers/infiniband/sw/rxe` compile against the OFED compat API
  (GPL-2.0, like the sources it patches):
  - drop the `struct ib_dmah *` parameter from `rxe_reg_user_mr`
    (upstream v6.16+ API; OFED's `ib_device_ops` predates it),
  - `umem_odp->map.pfn_list` → `umem_odp->pfn_list` (OFED keeps the
    pre-`hmm_dma_map` layout),
  - (via `rxe_ofed_compat.h`, injected by the makefile with `-include`)
    the two ODP capability bits OFED's enum lacks.
- `makefile` — external-module build against
  `/usr/src/ofa_kernel[-dkms]/<arch>/<kver>` (include chain +
  `KBUILD_EXTRA_SYMBOLS`, the same pattern MLNX's own iser/isert DKMS
  packages use). Refuses to build without an OFED tree: on inbox-rdma
  hosts the distro `modprobe rdma_rxe` is the right module.
- `prepare-src.sh` — sparse-clones the pinned upstream tag (default
  `v6.17`, image build-arg `rxe_kernel_ref`), applies the patch
  (`--fuzz=0`: drift fails the image build), and drops the makefile +
  compat header next to the sources.

`dev.Dockerfile` stages the patched source at `/opt/rxe-ofed/src` at image
build. The `.ko` itself is **per-host** and is built at container setup
time by `run_hw_ci.sh` (rxe mode mounts `/usr/src` + `/lib/modules`
read-only, builds in-container, and `insmod`s from the privileged
container). Nothing is loaded on hosts where the distro module works —
that is always tried first.

Gap 2 is fixed directly in `dev.Dockerfile`: it builds `librxe-rdmav<PABI>`
from the **same Mellanox rdma-core source release** as the installed
packages (the `MLNX_OFED_SRC-debian` bundle from the DOCA `SOURCES` repo,
sha256-pinned) and installs it into the image's libibverbs provider
directory.

## Caveats

- The loaded module does not survive a host reboot; the runner re-builds
  and re-loads on demand. For a permanent host install, DKMS-ify this
  directory (out of scope here).
- The pinned kernel ref should roughly match the host kernel generation;
  the OFED tree it compiles against, however, is whatever the host has
  installed. A very different host kernel will surface as loud compile
  errors at setup time (the lanes then SKIP, never silently).
- If a future DOCA release restores rxe (kernel or userspace), the distro
  `modprobe` / packaged provider win automatically and this machinery goes
  dormant.
