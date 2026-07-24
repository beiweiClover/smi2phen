"""Public Stage 05 network-proximity runner API with lazy scientific imports."""


def proximity_prepare_network(*args, **kwargs):
    from .prepare_network import proximity_prepare_network as implementation

    return implementation(*args, **kwargs)


def proximity_score_compounds(*args, **kwargs):
    from .score_compounds import proximity_score_compounds as implementation

    return implementation(*args, **kwargs)


__all__ = ["proximity_prepare_network", "proximity_score_compounds"]
