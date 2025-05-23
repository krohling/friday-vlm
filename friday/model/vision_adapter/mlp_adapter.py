import torch
import torch.nn as nn

class MLPAdapter(nn.Module):

    def __init__(self, input_dim, hidden_dim, output_dim, num_layers=2, activation='gelu', checkpoint_path=None, device=None, **kwargs):
        """
        Initialize the MLPAdapter with the given dimensions and activation function.

        Args:
            input_dim (int): Input dimension.
            hidden_dim (int): Hidden dimension.
            output_dim (int): Output dimension.
            layers (int): Number of layers in the MLP.
            activation (str): Activation function to use ('gelu' or 'relu').
        """
        super().__init__()
        self.num_layers = num_layers
        self.activation = activation
        self.output_dim = output_dim

        # Define the first layer
        layers_list = [nn.Linear(input_dim, hidden_dim, device=device)]
        if activation == 'gelu':
            layers_list.append(nn.GELU())
        elif activation == 'relu':
            layers_list.append(nn.ReLU())
        else:
            raise ValueError("Unsupported activation function. Use 'gelu' or 'relu'.")
        
        # Define the subsequent layers
        for _ in range(1, num_layers):
            layers_list.append(nn.Linear(hidden_dim, hidden_dim, device=device))
            if activation == 'gelu':
                layers_list.append(nn.GELU())
            elif activation == 'relu':
                layers_list.append(nn.ReLU())
        
        # Define the final output layer
        layers_list.append(nn.Linear(hidden_dim, output_dim, device=device))
        self.mlp = nn.Sequential(*layers_list)

        # Load checkpoint if provided
        if checkpoint_path:
            self.load_state_dict(torch.load(checkpoint_path, map_location=device), strict=False)
            print(f"Loaded MLPAdapter from {checkpoint_path}")
        
        if device:
            self.to(device)

    def forward(self, x):
        """
        Forward pass through the MLPAdapter.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output tensor after passing through the MLP.
        """
        return self.mlp(x)