__all__ = ["mla_prolog", "mla_prolog_v3"]

from .mla_prolog import mla_prolog

# Wheel discovery uses the operator directory name; retain the public alias.
mla_prolog_v3 = mla_prolog
