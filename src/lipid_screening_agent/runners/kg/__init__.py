"""Public KG runner API with lazy scientific imports."""


def kg_construct_graph(*args, **kwargs):
    from .construct_graph import kg_construct_graph as implementation

    return implementation(*args, **kwargs)


def kg_prepare_training_data(*args, **kwargs):
    from .prepare_training_data import kg_prepare_training_data as implementation

    return implementation(*args, **kwargs)


def kg_pretrain(*args, **kwargs):
    from .pretrain import kg_pretrain as implementation

    return implementation(*args, **kwargs)


def kg_finetune_seed(*args, **kwargs):
    from .finetune_seed import kg_finetune_seed as implementation

    return implementation(*args, **kwargs)


def kg_aggregate_seeds(*args, **kwargs):
    from .aggregate_seeds import kg_aggregate_seeds as implementation

    return implementation(*args, **kwargs)


__all__ = [
    "kg_aggregate_seeds",
    "kg_construct_graph",
    "kg_finetune_seed",
    "kg_prepare_training_data",
    "kg_pretrain",
]
