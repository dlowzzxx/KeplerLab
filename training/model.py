"""
model.py - Physics-Informed Neural Network architecture.

Architecture overview:
    t (scalar) -> Fourier Feature Embedding -> MLP (4x64 tanh) -> [x, y, vx, vy]

Key design decisions:

1. FOURIER FEATURES (integer harmonics)
   In normalized coordinates one orbit = 2pi, so the state vector is periodic
   with period 2pi.  We embed the scalar time into sin/cos pairs at integer
   multiples of the fundamental frequency:
       [sin(t), cos(t), sin(2t), cos(2t), ..., sin(Nt), cos(Nt)]
   This lets the MLP represent arbitrary Fourier series, and by omitting
   raw t the output is *exactly* periodic by construction - which is the
   correct physical constraint for unperturbed Keplerian orbits.

2. TANH ACTIVATION
   The physics loss requires d^2output/dt^2 via autograd.  Tanh is Cinf
   (infinitely differentiable) so higher-order gradients are well-defined
   everywhere.  ReLU has a discontinuous first derivative at 0 and zero
   second derivative everywhere else - it would break the ODE residuals.

3. NETWORK SIZE
   4 hidden layers x 64 neurons ~ 14,000 parameters.  This is deliberately
   small: the physics loss provides such a strong inductive bias that a
   compact network suffices.  It also makes browser inference trivial.

4. XAVIER INITIALIZATION
   Matches the tanh activation's linear regime, preventing gradient
   vanishing/explosion at initialization.
"""

import torch
import torch.nn as nn


class FourierFeatures(nn.Module):
    """
    Sinusoidal positional encoding using integer harmonics.

    Maps scalar time t to a 2N-dimensional vector:
        [sin(1*t), cos(1*t), sin(2*t), cos(2*t), ..., sin(N*t), cos(N*t)]

    This is the natural basis for periodic functions with period 2pi.
    For elliptical orbits the motion contains higher harmonics of the
    orbital frequency, so N=10 captures up to the 10th overtone.
    """

    def __init__(self, num_frequencies: int = 10):
        super().__init__()
        self.num_frequencies = num_frequencies
        # Integer harmonics: 1, 2, 3, ..., N
        freqs = torch.arange(1, num_frequencies + 1, dtype=torch.float32)
        self.register_buffer("frequencies", freqs)

    @property
    def output_dim(self) -> int:
        """Dimensionality of the feature vector (2N: sin + cos)."""
        return 2 * self.num_frequencies

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        """
        Args:
            t: Normalized time, shape (batch, 1)
        Returns:
            Fourier features, shape (batch, 2*num_frequencies)
        """
        # t shape: (batch, 1), frequencies shape: (N,)
        # Broadcasting: (batch, 1) * (N,) -> (batch, N)
        args = t * self.frequencies
        return torch.cat([torch.sin(args), torch.cos(args)], dim=-1)


class PINN(nn.Module):
    """
    Physics-Informed Neural Network for Keplerian orbital dynamics.

    Takes normalized time t as input and outputs the normalized state
    vector [x, y, vx, vy].  All physics constraints are enforced
    through the loss function, not the architecture itself.

    Parameters:
        hidden_dim:       Width of each hidden layer
        hidden_layers:    Number of hidden layers
        num_fourier_freq: Number of Fourier embedding frequencies
    """

    def __init__(self, hidden_dim: int = 64, hidden_layers: int = 4,
                 num_fourier_freq: int = 10):
        super().__init__()

        self.fourier = FourierFeatures(num_fourier_freq)

        # Build MLP: input -> [hidden -> tanh]xN -> output
        layers = []
        in_dim = self.fourier.output_dim

        for _ in range(hidden_layers):
            layers.append(nn.Linear(in_dim, hidden_dim))
            layers.append(nn.Tanh())
            in_dim = hidden_dim

        # Output layer: 4 components [x, y, vx, vy] - no activation
        layers.append(nn.Linear(hidden_dim, 4))

        self.network = nn.Sequential(*layers)
        self._initialize_weights()

    def _initialize_weights(self):
        """Xavier normal initialization - optimal for tanh networks."""
        for module in self.network:
            if isinstance(module, nn.Linear):
                nn.init.xavier_normal_(module.weight)
                nn.init.zeros_(module.bias)

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        """
        Forward pass: time -> state.

        Args:
            t: Normalized time, shape (batch, 1)
        Returns:
            state: [x, y, vx, vy] in normalized coordinates, shape (batch, 4)
        """
        features = self.fourier(t)
        return self.network(features)

    def count_parameters(self) -> int:
        """Total number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
