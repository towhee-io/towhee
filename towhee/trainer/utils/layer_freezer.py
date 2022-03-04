from typing import Union


class LayerFreezer:
    """
    Utilities to freeze/unfreeze layers.

    Args:
        model:
            a model with weights.
    Return:
        LayerFreezer.

    Example:
        >>> from towhee.trainer.utils.layer_freezer import LayerFreezer
        >>> from towhee.models.vit.vit import VitModel
        >>> my_model = VitModel('vit_base_patch16_224')
        >>> my_freezer = LayerFreezer(my_model)
        >>> # Check if modules in the last layer are frozen
        >>> my_freezer.status(-1)
        >>> # Check if modules in the layer "head" are frozen
        >>> my_freezer.status("head")
        ['unfrozen', 'unfrozen']
        >>> # Show all frozen layers
        >>> my_freezer.show_frozen_layers()
        ['patch_embed', 'head']
        >>> # Freeze layers by a list of layer indexes
        >>> my_freezer.by_idx([0, -1])
        >>> # Freeze layers by a list of layer names
        >>> my_freezer.by_names(['head'])
        >>> # Freeze all layers
        >>> my_freezer.set_all()
        >>> # Unfreeze all layers
        >>> my_freezer.set_all(freeze=False)
        >>> # Freeze all except the last layer
        >>> my_freezer.set_slice(-1)
    """

    def __init__(self, model):
        self.model = model
        self.layer_names = list(dict(self.model.named_children()).keys())
        self.layer_count = len(self.layer_names)

    def status(self, layer: Union[str, int]):
        """
        Check if a layer is frozen or not by its name or index

        Args:
            layer (`Union[str, int]`):
                the name or index of layer.
        Return:
            A list of status ('frozen' or 'unfrozen') to indicate if modules in the layer are frozen or not.
        """
        if isinstance(layer, str):
            if layer not in self.layer_names:
                raise ValueError(f'Layer does not exist: {layer}')
            for name, child in self.model.named_children():
                if name == layer:
                    outs = []
                    for param in child.parameters():
                        outs.append('unfrozen' if param.requires_grad else 'frozen')
                    print(f'Freeze status of layer {layer}: {outs}')
                    return outs
        if isinstance(layer, int):
            outs = []
            for param in list(self.model.children())[layer].parameters():
                outs.append('unfrozen' if param.requires_grad else 'frozen')
            print(f'Freeze status of layer {layer}: {outs}')
            return outs

    def show_frozen_layers(self):
        """
        Show all names of frozen layers

        Args:
            None.
        Return:
            A list of names of frozen layers
        """
        outs = []
        for name, layer in self.model.named_children():
            flag = 0
            for param in layer.parameters():
                if not param.requires_grad:
                    flag = 1
            if flag == 1:
                outs.append(name)
        # print(f'{len(outs)} frozen layers (names): {outs}')
        return outs

    def by_names(self, names: list, freeze: bool = True):
        """
        Freeze/unfreeze layers by names

        Args:
            names (`list`):
                a list of layer names
            freeze (`bool`):
                if or not freeze layers (default: True)
        """
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
        """
        Freeze/unfreeze layers by indexes

        Args:
            idx (`list`):
                a list of layer indexes
            freeze (`bool`):
                if or not freeze layers (default: True)
        """
        for i in idx:
            for param in list(self.model.children())[i].parameters():
                param.requires_grad = not freeze

    def set_all(self, freeze: bool = True):
        """
        Freeze/unfreeze all layers.

        Args:
            freeze (`bool`):
                if or not freeze layers (default: True)
        """
        for layer in self.model.children():
            for param in layer.parameters():
                param.requires_grad = not freeze

    def set_slice(self, slice_num: int, freeze: bool = True):
        """
        Freeze/unfreeze layers by list slice.

        Args:
            slice_num (`int`):
                number to slice the list of layers
            freeze (`bool`):
                if or not freeze layers (default: True)
        """
        myslice = slice(slice_num)
        slice_idx = list(range(self.layer_count))[myslice]
        self.by_idx(idx=slice_idx, freeze=freeze)
