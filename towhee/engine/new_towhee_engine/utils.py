def flatten(tpl):
    if isinstance(tpl, (tuple, list)):
        for v in tpl:
            yield from flatten(v)
    else:
        yield tpl