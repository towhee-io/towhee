class LayerFreezer:
    """
    Freeze/unfreeze layers by names or index.
    Args:
        model: model with weights
    Example usage:
        freezer = LayerFreezer(model) # Setup freezer with model
        freezer.status('classifier') # Check status of the "classifier" layer
        freezer.status(-1) # Check status of the last layer
        freezer.show_freezed_layers() # Show names of freezed layers
        freezer.set_all() # Freeze all layers
        freezer.set_slice(3) # Freeze first 3 layers
        freezer.by_name(['norm', 'classifier']) # Freeze "norm" layer and "classifier" layer
        freezer.by_idx([-1, -2]) # Freeze last two layers
    """

    def __init__(self, model):
        self.model = model
        self.layer_names = list(dict(self.model.named_children()).keys())
        self.layer_count = len(self.layer_names)

    def status(self, layer: str or int):
        if isinstance(layer, str):
            if layer not in self.layer_names:
                raise ValueError(f'Layer does not exist: {layer}')
            for name, child in self.model.named_children():
                if name == layer:
                    outs = []
                    for param in child.parameters():
                        outs.append('unfreezed' if param.requires_grad else 'freezed')
                    print(f'Freeze status of layer {layer}: {outs}')
                    return outs
        if isinstance(layer, int):
            outs = []
            for param in list(self.model.children())[layer].parameters():
                outs.append('unfreezed' if param.requires_grad else 'freezed')
            print(f'Freeze status of layer {layer}: {outs}')
            return outs

    def show_freezed_layers(self):
        outs = []
        for name, layer in self.model.named_children():
            flag = 0
            for param in layer.parameters():
                if not param.requires_grad:
                    flag = 1
            if flag == 1:
                outs.append(name)
        print(f'{len(outs)} freezed layers (names): {outs}')
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

    def set_slice(self, slice_num: int, freeze: bool = True):
        myslice = slice(slice_num)
        slice_idx = list(range(self.layer_count))[myslice]
        self.by_idx(idx=slice_idx, freeze=freeze)
