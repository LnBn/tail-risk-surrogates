"""tailsurr: constrained surrogates for financial tail-risk analysis."""

from .fitting import fit_mle, sample_family, select_xmin, sf, umvue_sf, vuong_test
from .surrogates import (FAMILIES, constrained_surrogates, default_n_steps,
                         shuffle_surrogates, typical_surrogates)
from .stats import (ALTERNATIVE, sample_max, surrogate_pvalue, tail_statistics,
                    top_hill, waiting_time_cv)
