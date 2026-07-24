"""Final candidate ranking and run-report runners."""


def rank_candidates(*args, **kwargs):
    from .rank_candidates import rank_candidates as implementation

    return implementation(*args, **kwargs)


def compute_candidate_ranking(*args, **kwargs):
    from .rank_candidates import compute_candidate_ranking as implementation

    return implementation(*args, **kwargs)


def rank_percentiles(*args, **kwargs):
    from .rank_candidates import rank_percentiles as implementation

    return implementation(*args, **kwargs)


def generate_run_report(*args, **kwargs):
    from .generate_run_report import generate_run_report as implementation

    return implementation(*args, **kwargs)


__all__ = [
    "compute_candidate_ranking",
    "generate_run_report",
    "rank_candidates",
    "rank_percentiles",
]
