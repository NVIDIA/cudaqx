# SPDX-License-Identifier: GPL-2.0
#
# Build the (patched) upstream rxe SoftRoCE driver as an external module
# against the MLNX/DOCA-OFED ofa_kernel compat headers + Module.symvers, so
# it binds to the OFED DKMS ib_core instead of the in-tree one.
#
# Why: on hosts with doca-ofed / mlnx-ofed-kernel-dkms installed (e.g.
# GB200), ib_core & friends come from /lib/modules/*/updates/dkms and export
# different symbol CRCs than the in-tree build, so the distro's rdma_rxe.ko
# fails to load with "disagrees about version of symbol ib_*" -- and OFED
# dropped the rxe driver from its own source tree.  The external-module
# pattern below (OFED include chain + KBUILD_EXTRA_SYMBOLS) is copied from
# the makefiles of MLNX OFED's own out-of-tree DKMS packages (iser/isert).
#
# This makefile, rxe_ofed_compat.h and ofed-compat.patch are expected to sit
# next to (and the patch already applied to) the upstream
# drivers/infiniband/sw/rxe sources -- prepare-src.sh stages exactly that.
# GNU make prefers this lowercase 'makefile'; kbuild's sub-make reads the
# upstream capital 'Makefile' for the object list.

KVER ?= $(shell uname -r)
ARCH_M := $(shell uname -m)
OFA_DIR ?= $(shell test -d /usr/src/ofa_kernel-dkms/$(ARCH_M) && echo /usr/src/ofa_kernel-dkms/$(ARCH_M) || echo /usr/src/ofa_kernel/$(ARCH_M))
OFA := $(OFA_DIR)/$(KVER)
K_BUILD ?= /lib/modules/$(KVER)/build

ifneq ($(wildcard $(OFA)/Module.symvers),)
include $(OFA)/configure.mk.kernel
endif

autoconf_h := $(shell /bin/ls -1 $(K_BUILD)/include/*/autoconf.h 2>/dev/null | head -1)
kconfig_h := $(shell /bin/ls -1 $(K_BUILD)/include/*/kconfig.h 2>/dev/null | head -1)
ifneq ($(kconfig_h),)
KCONFIG_H = -include $(kconfig_h)
endif

LINUXINCLUDE=\
	-include $(autoconf_h) \
	$(KCONFIG_H) \
	-include $(OFA)/include/linux/compat-2.6.h \
	-include $(CURDIR)/rxe_ofed_compat.h \
	-I$(OFA)/include \
	-I$(OFA)/include/uapi \
	$(BACKPORT_INCLUDES) \
	-I$$(srctree)/arch/$$(SRCARCH)/include \
	-I$$(objtree)/arch/$$(SRCARCH)/include/generated \
	-I$$(objtree)/include \
	-I$$(srctree)/arch/$$(SRCARCH)/include/uapi \
	-I$$(objtree)/arch/$$(SRCARCH)/include/generated/uapi \
	-I$$(srctree)/include \
	-I$$(srctree)/include/uapi \
	-I$$(objtree)/include/generated/uapi \
	#

default:
ifneq ($(wildcard $(OFA)/Module.symvers),)
	$(MAKE) -C $(K_BUILD) O=$(K_BUILD) M=$(CURDIR) \
		KBUILD_EXTRA_SYMBOLS=$(OFA)/Module.symvers \
		CONFIG_RDMA_RXE=m \
		LINUXINCLUDE='$(LINUXINCLUDE)' \
		modules
else
	@echo "error: $(OFA_DIR)/$(KVER) has no Module.symvers." >&2; \
	echo "This OFED-compat build is only for MLNX/DOCA-OFED DKMS hosts;" >&2; \
	echo "on inbox-rdma hosts load the distro module: sudo modprobe rdma_rxe" >&2; \
	echo "(if run from the hw_ci container: are /usr/src and /lib/modules" >&2; \
	echo "mounted from the host?)" >&2; \
	exit 1
endif

clean:
	rm -f *.o *.ko *.mod *.mod.c modules.order Module.symvers .*.cmd
