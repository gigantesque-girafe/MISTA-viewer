"""Deterministic IK retarget (the `--retarget` "B arm"): no-op

"""

_WARNED = False


def add_retarget_args(parser):
    """Register the `--retarget` CLI group. Defaults keep today's (retarget-off) behavior."""
    g = parser.add_argument_group("retarget (not yet implemented — passthrough only)")
    g.add_argument("--retarget", action="store_true",
                   help="Enable IK retarget. NOT IMPLEMENTED: poses pass through unmodified.")
    g.add_argument("--source-jtr-npz", type=str, default=None,
                   help="Path to an .npz with a 'Jtr' array: the actor rest skeleton, "
                        "known up front instead of locked from live betas.")
    g.add_argument("--proportion", type=float, default=1.0,
                   help="Proportion retarget blend factor (unused: retarget is a no-op).")
    g.add_argument("--source-betas-frames", type=int, default=30,
                   help="Number of live betas frames to average before locking the actor "
                        "rest skeleton, when --source-jtr-npz is not given.")
    return parser


def build_retargeter(args, Jtr_target, source_Jtr):
    """Always returns None: IK retarget is not implemented, so poses pass through unmodified.

    @note: Warns once if the caller actually requested `--retarget`, so opting in
        isn't silently misleading.
    """
    global _WARNED
    if getattr(args, "retarget", False) and not _WARNED:
        print("[WARN] --retarget requested but IK retarget is not implemented yet; "
              "poses will pass through unmodified.", flush=True)
        _WARNED = True
    return None


def source_jtr_from_betas(betas_mean):
    """Inert placeholder: IK retarget is not implemented, so no source skeleton is derived."""
    return None
