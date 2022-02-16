class LayerFreezer:
    """
    Freeze/unfreeze layers by names or index.
    Args:
        model: model with weights
    Example usage:
        freezer = LayerFreezer(model)
        freezer.status('classifier')
        freezer.status(-1)
        freezer.set_all()
        freezer.by_name(['classifier'])
        freezer.by_idx([-1])
    """

    def __init__(self, model):
        self.model = model
        self.layer_names = list(dict(self.model.named_children()).keys())

    def status(self, layer: str or int):
        if isinstance(layer, str):
            if layer not in self.layer_names:
                raise ValueError(f'Layer does not exist: {layer}')
            for name, child in self.model.named_children():
                if name == layer:
                    outs = []
                    for param in child.parameters():
                        outs.append('unfreezed' if param.requires_grad else 'freezed')
                    return outs
        if isinstance(layer, int):
            outs = []
            for param in list(self.model.children())[layer].parameters():
                outs.append('unfreezed' if param.requires_grad else 'freezed')
            return outs

    def by_names(self, names: list, freeze: bool = True):
        if not set(names).issubset(set(self.layer_names)):
            invalid_names = set(names) - set(self.layer_names)
            raise ValueError(f'Layer does not exist: {invalid_names}')
        for name, layer in self.model.named_children():
            if name not in names:
                pass
            else:
                for param in layer.parameters():
                    param.requires_grad = not freeze

    def by_idx(self, idx: list, freeze: bool = True):
        for i in idx:
            for param in list(self.model.children())[i].parameters():
                param.requires_grad = not freeze

    def set_all(self, freeze: bool = True):
        for layer in self.model.children():
            for param in layer.parameters():
                param.requires_grad = not freeze
