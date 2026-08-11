/* SPDX-License-Identifier: GPL-2.0 */
/*
 * The MLNX/DOCA-OFED compat enum ib_odp_transport_cap_bits ends at
 * IB_ODP_SUPPORT_SRQ_RECV (1 << 5); these two bits exist upstream in v6.17
 * (values mirror uapi IB_UVERBS_ODP_SUPPORT_* exactly).  Injected via
 * -include from the accompanying makefile so the upstream rxe sources need
 * no edit for this.
 */
#ifndef RXE_OFED_COMPAT_H
#define RXE_OFED_COMPAT_H

#define IB_ODP_SUPPORT_FLUSH (1 << 6)
#define IB_ODP_SUPPORT_ATOMIC_WRITE (1 << 7)

#endif
