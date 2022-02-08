from torch.utils import model_zoo
import torch
import logging

def load_pretrained_weights(
    model,
    model_name=None,
    model_configs=None,
    weights_path=None,
):
    """
    Load pretrained weights from path or url.
    Args:
        model(nn.Module):
            Model.
        model_name(str):
            Model name
        model_configs(dict):
            Model configs
        weights_path(str, optional):
            Path to pretrained weights file on the local disk.
        verbose (bool):
            If printing is done when downloading is over.
    """
    logging.info('model_name is %s', str(model_name))
    logging.info('weights_path is %s', str(weights_path))
    assert bool(model_name), 'Expected model_name'

    # Load or download weights
    if weights_path is None:
        logging.info(model_configs.keys())
        url = model_configs['url']
        if url:
            logging.info('Please check hub connection in case weights can not be downloaded!')
            state_dict = model_zoo.load_url(url, map_location=torch.device('cpu'))
        else:
            raise ValueError(f'Pretrained model for {model_name} is not available.')
    else:
        state_dict = torch.load(weights_path)

   # Load state dict
    if 'model' in state_dict:
        ret = model.load_state_dict(state_dict['model'], strict=False)
    else:
        ret = model.load_state_dict(state_dict, strict=False)
    return ret
