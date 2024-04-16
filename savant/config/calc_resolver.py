"""Calculator resolver for OmegaConf."""

from simpleeval import simple_eval


def calc_resolver(calc_string: str, *args):
    """OmegaConf resolver that provides config variable value by evaluating an
    arithmetic expression passed by string representation. It's possible to
    refer to custom arguments in the expression with names like "arg_x", where
    `x` is the argument index.

    Example usage in config:

    .. code-block:: yaml

        parameters:
            sum: ${calc:"1+1"}
            base: 100
            derived: ${calc:"arg_0*arg_1",${.base},0.5}

    where

    * ``calc`` is the registered resolver name
    * ``sum`` doesn't depend on other arguments and evaluates to 2
    * ``derived`` depends on ``base`` parameter value and evaluates to 50
    """
    names = {f'arg_{i}': arg for i, arg in enumerate(args)}
    return simple_eval(calc_string, names=names)
