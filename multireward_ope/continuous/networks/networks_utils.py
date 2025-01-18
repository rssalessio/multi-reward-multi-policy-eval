import torch.nn as nn
from multireward_ope.continuous.networks.ensemble_linear_layer import EnsembleLinear


def make_single_network(input_size: int, output_size: int, hidden_size: int, ensemble_size: int, final_activation = nn.ReLU) -> nn.Module:
    """ Create a single network """
    net = [
        EnsembleLinear(input_size, hidden_size, ensemble_size) if ensemble_size > 1 else nn.Linear(input_size, hidden_size),
        nn.ReLU(),
        EnsembleLinear(hidden_size, hidden_size, ensemble_size) if ensemble_size > 1 else nn.Linear(hidden_size, hidden_size),
        nn.ReLU(),
        EnsembleLinear(hidden_size, output_size, ensemble_size) if ensemble_size > 1 else nn.Linear(hidden_size, output_size)]
    if final_activation is not None:
        net.append(final_activation())
    
    return nn.Sequential(*net)
