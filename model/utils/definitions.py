"""
definitions.py

Author: T. Grossi
E-mail: tommaso.grossi@ing.unipi.it
Year: 2025

Class definitions for machine-learning scripts
"""

# Import necessary libraries
import torch
import torch.nn as nn


# Class that loads the entire dataset on the GPU, and that defines an iterator to perform the training
# in random batches of size batch_size
# In addition, it randomly shuffles the dataset at the beginning of an iteration
class FEMDataset:
    def __init__(
        self,
        input_tensor,
        target_tensor,
        branch_data_tensor,
        nomad_parameter_tensor,
        analysis_data_indexes,
        analysis_data_factor,
        batch_size,
        device,
    ):
        if input_tensor is not None:
            self.input_tensor = input_tensor.to(device)
            self.coordinates_data = True
        else:
            self.input_tensor = None
            self.coordinates_data = False
        self.target_tensor = target_tensor.to(device)
        self.branch_data_tensor = branch_data_tensor.to(device)
        self.nomad_parameter_tensor = nomad_parameter_tensor.to(device)

        # Check whether analysis_data_factor is complex (i.e., we are training a SignNet with FY and MY of varying signs)
        if (
            analysis_data_factor.dtype == torch.complex64
            or analysis_data_factor.dtype == torch.complex128
        ):
            # Use the same dtype as analysis_data_indexes
            self.analysis_data_factor = torch.zeros(
                (analysis_data_factor.shape[0], branch_data_tensor.shape[1]),
                dtype=target_tensor.dtype,
                device=device,
            )
            # The first two/thirds of the columns are the real part of the factor, the last third is the imaginary part
            self.analysis_data_factor[:, : 2 * branch_data_tensor.shape[1] // 3] = (
                torch.broadcast_to(
                    analysis_data_factor.unsqueeze(1).real,
                    (
                        len(analysis_data_factor),
                        2 * branch_data_tensor.shape[1] // 3,
                    ),
                )
            )
            self.analysis_data_factor[:, 2 * branch_data_tensor.shape[1] // 3 :] = (
                torch.broadcast_to(
                    analysis_data_factor.unsqueeze(1).imag,
                    (len(analysis_data_factor), branch_data_tensor.shape[1] // 3),
                )
            )
        else:
            self.analysis_data_factor = analysis_data_factor.unsqueeze(1).to(device)

        self.analysis_data_indexes = analysis_data_indexes.to(device)
        self.num_samples = self.target_tensor.shape[0]
        self.device = device
        self.batch_size = batch_size

    def __len__(self):
        return self.num_samples // self.batch_size + 1

    def __iter__(self):
        self.indices = torch.randperm(self.num_samples)
        self.i = 0
        return self

    def __next__(self):
        if self.i >= self.num_samples:
            raise StopIteration
        else:
            batch_indices = self.indices[self.i : self.i + self.batch_size]
            self.i += self.batch_size

            if self.coordinates_data:
                # The first column of the input tensor is the index of the analysis data tensor to prepend to the input tensor
                # The second column of the input tensor is the multiplicative factor to apply to the analysis data tensor
                return (
                    torch.cat(
                        (
                            self.branch_data_tensor[
                                self.analysis_data_indexes[batch_indices]
                            ]
                            * self.analysis_data_factor[batch_indices],
                            self.nomad_parameter_tensor[
                                self.analysis_data_indexes[batch_indices]
                            ],
                            self.input_tensor[batch_indices],
                        ),
                        dim=1,
                    ),
                    self.target_tensor[batch_indices],
                )
            else:
                return (
                    torch.cat(
                        (
                            self.branch_data_tensor[
                                self.analysis_data_indexes[batch_indices]
                            ]
                            * self.analysis_data_factor[batch_indices],
                            self.nomad_parameter_tensor[
                                self.analysis_data_indexes[batch_indices]
                            ],
                        ),
                        dim=1,
                    ),
                    self.target_tensor[batch_indices],
                )


# Branch feed-forward neural network
class BranchNet(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, hidden_layers, activation):
        super(BranchNet, self).__init__()
        self.fc = nn.ModuleList([nn.Linear(input_dim, hidden_dim)])
        for i in range(hidden_layers):
            self.fc.append(nn.Linear(hidden_dim, hidden_dim))
        self.fc.append(nn.Linear(hidden_dim, output_dim))
        self.activation = activation

    def forward(self, x):
        for i in range(len(self.fc) - 1):
            x = self.activation(self.fc[i](x))
        x = self.fc[-1](x)
        return x


# NonLinear Manifold Decoder (NOMAD) feed-forward neural network
class NOMAD(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, hidden_layers, activation):
        super(NOMAD, self).__init__()
        self.fc = nn.ModuleList([nn.Linear(input_dim, hidden_dim)])
        for i in range(hidden_layers):
            self.fc.append(nn.Linear(hidden_dim, hidden_dim))
        self.fc.append(nn.Linear(hidden_dim, output_dim))
        self.activation = activation

    def forward(self, x):
        for i in range(len(self.fc) - 1):
            x = self.activation(self.fc[i](x))
        x = self.fc[-1](x)
        return x


# Class that defines the NeuberNet model for a single target variable
class NeuberNetComponent(nn.Module):
    def __init__(
        self,
        branch_input_dim,
        nomad_secondary_input_dim,
        nomad_output_dim,
        branch_hidden_dim,
        nomad_hidden_dim,
        branch_hidden_layers,
        nomad_hidden_layers,
        n_terms,
        activation,
    ):
        super(NeuberNetComponent, self).__init__()

        self.branch_input_dim = branch_input_dim
        self.n_terms = n_terms
        self.nomad_secondary_input_dim = nomad_secondary_input_dim
        self.nomad_input_dim = self.n_terms + self.nomad_secondary_input_dim
        self.nomad_output_dim = nomad_output_dim
        self.branch_hidden_dim = branch_hidden_dim
        self.nomad_hidden_dim = nomad_hidden_dim
        self.branch_hidden_layers = branch_hidden_layers
        self.nomad_hidden_layers = nomad_hidden_layers
        self.activation = activation

        self.branch_net = BranchNet(
            self.branch_input_dim,
            self.branch_hidden_dim,
            self.n_terms,
            self.branch_hidden_layers,
            self.activation,
        )
        self.nomad = NOMAD(
            self.n_terms + self.nomad_secondary_input_dim,
            self.nomad_hidden_dim,
            self.nomad_output_dim,
            self.nomad_hidden_layers,
            self.activation,
        )

    def forward(self, x):
        branch_input = x[:, : self.branch_input_dim]
        branch_output = self.branch_net(branch_input)
        nomad_input = torch.cat(
            (
                branch_output,
                x[:, self.branch_input_dim :],
            ),
            dim=1,
        )
        return self.nomad(nomad_input)


# Class that defines a general ensemble of NeuberNet components
class NeuberNetEnsemble(nn.Module):
    def __init__(
        self,
        components,
        normalize=False,
        input_mean=None,
        input_std=None,
        output_mean=None,
        output_std=None,
    ):
        super(NeuberNetEnsemble, self).__init__()

        self.components = nn.ModuleList(components)
        self.normalize = normalize
        self.register_buffer("input_mean", input_mean)
        self.register_buffer("input_std", input_std)
        self.register_buffer("output_mean", output_mean)
        self.register_buffer("output_std", output_std)

    def forward(self, x):
        if self.normalize:
            x = (x - self.input_mean) / self.input_std
            x = torch.cat([component(x) for component in self.components], dim=1)
            return x * self.output_std + self.output_mean
        else:
            return torch.cat([component(x) for component in self.components], dim=1)


# Class that defines the YieldNet model
class YieldNet(NeuberNetEnsemble):
    def __init__(
        self,
        components,
        normalize=False,
        input_mean=None,
        input_std=None,
        output_mean=None,
        output_std=None,
    ):
        super(YieldNet, self).__init__(
            components,
            normalize,
            input_mean,
            input_std,
            output_mean,
            output_std,
        )

    def forward(self, x):
        if self.normalize:
            x = (x - self.input_mean) / self.input_std
            x = torch.cat([component(x) for component in self.components], dim=1)
            x = x * self.output_std + self.output_mean
            return x
        else:
            return torch.cat([component(x) for component in self.components], dim=1)


# Class that defines the SignNet model
class SignNet(NeuberNetEnsemble):
    def __init__(
        self,
        components,
        normalize=False,
        input_mean=None,
        input_std=None,
        output_mean=None,
        output_std=None,
    ):
        super(SignNet, self).__init__(
            components,
            normalize,
            input_mean,
            input_std,
            output_mean,
            output_std,
        )

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        if self.normalize:
            x = (x - self.input_mean) / self.input_std
            x = torch.cat([component(x) for component in self.components], dim=1)
            return self.sigmoid(x)
        else:
            # We train the model with logits, not probabilities
            return torch.cat([component(x) for component in self.components], dim=1)


# Class that defines the NeuberNet model for all the target variables
class NeuberNet(NeuberNetEnsemble):
    def __init__(
        self,
        components,
        normalize=False,
        input_mean=None,
        input_std=None,
        output_mean=None,
        output_std=None,
        production_mode=False,
        yieldnet=None,
        signnet=None,
        yield_tol=2.5e-2,
        small_scale_pl_tol=0.25,
    ):
        super(NeuberNet, self).__init__(
            components,
            normalize,
            input_mean,
            input_std,
            output_mean,
            output_std,
        )
        self.branch_input_dim = components[0].branch_input_dim
        self.production_mode = production_mode
        self.yieldnet = yieldnet
        self.signnet = signnet
        self.yield_tol = yield_tol
        self.small_scale_pl_tol = small_scale_pl_tol

    def forward(self, x):
        if self.production_mode:
            if self.yieldnet is not None and self.signnet is not None:
                # Unsqueeze x to add a batch dimension, if it is not already present
                if len(x.shape) == 1:
                    x = x.unsqueeze(0)

                # Prepare normalized branch input for SignNet and YieldNet
                branch_input = x[:, : self.branch_input_dim]
                norm_branch_input = torch.sqrt(
                    torch.linalg.vector_norm(
                        branch_input[:, : 2 * (self.branch_input_dim // 3)],
                        dim=-1,
                        keepdim=True,
                    )
                    ** 2
                    + torch.linalg.vector_norm(
                        branch_input[:, 2 * (self.branch_input_dim // 3) :]
                        * x[
                            :, [self.branch_input_dim]
                        ],  # Rotations are multiplied by R for homogeneity with displacements
                        dim=-1,
                        keepdim=True,
                    )
                    ** 2
                )
                aux_x = x.detach().clone()[:, :-2]  # Inputs without spatial coordinates
                aux_x[
                    :, : self.branch_input_dim
                ] /= norm_branch_input  # Normalized branch input
                aux_x.nan_to_num()

                # Propagate through SignNet
                sign_x = self.signnet(aux_x)
                sign_x = torch.sign(sign_x - 0.5)

                # Multiply the first 2/3 of branch data by sign_x[..., 0] and the last 1/3 by sign_x[..., 1]
                x[:, : 2 * (self.branch_input_dim // 3)] *= sign_x[:, [0]]
                x[
                    :, 2 * (self.branch_input_dim // 3) : self.branch_input_dim
                ] *= sign_x[:, [1]]
                aux_x[:, : 2 * (self.branch_input_dim // 3)] *= sign_x[:, [0]]
                aux_x[
                    :, 2 * (self.branch_input_dim // 3) : self.branch_input_dim
                ] *= sign_x[:, [1]]

                yield_x = self.yieldnet(aux_x)
                yield_x[:, 0] *= norm_branch_input[
                    :, 0
                ]  # Find elastic von mises stress for denormalized branch inputs
                scaling_factors = torch.zeros(yield_x.shape[0], device=yield_x.device)
                # When 0 < yield_x[:, 0] <= 1, scaling_factors are 1/yield_x[:, 0]
                # Else, scaling_factors are simply 1
                scaling_factors[
                    (yield_x[:, 0] > 0) & (yield_x[:, 0] <= 1 + self.yield_tol)
                ] = (
                    1
                    / yield_x[
                        (yield_x[:, 0] > 0) & (yield_x[:, 0] <= 1 + self.yield_tol), 0
                    ]
                )
                scaling_factors[yield_x[:, 0] > 1 + self.yield_tol] = 1

                # ### Strict version, which set all outputs to NaNs if small-scale plasticity is supposedly violated ###
                # # When 1 + self.yield_tol < yield_x[:, 0] <= yield_x[:, 1], scaling_factors are 1
                # # When yield_x[:, 1] + self.small_scale_pl_tol < yield_x[:, 0], scaling_factors are NaNs
                # scaling_factors[(yield_x[:, 0] > 0) & (yield_x[:, 0] <= 1 + self.yield_tol)] = (
                #     1 / yield_x[(yield_x[:, 0] > 0) & (yield_x[:, 0] <= 1 + self.yield_tol), 0]
                # )
                # scaling_factors[
                #     (yield_x[:, 0] > 1 + self.yield_tol)
                #     & (yield_x[:, 0] <= yield_x[:, 1] + self.small_scale_pl_tol)
                # ] = 1
                # scaling_factors[
                #     yield_x[:, 1] + self.small_scale_pl_tol < yield_x[:, 0]
                # ] = float("nan")

                # Multiply the branch data by the scaling factors
                x[:, : self.branch_input_dim] *= scaling_factors.unsqueeze(1)

                # Propagate through the NeuberNet ensemble
                x = (x - self.input_mean) / self.input_std
                x = torch.cat([component(x) for component in self.components], dim=1)
                x = x * self.output_std + self.output_mean

                # Rescale the results by the scaling factors
                x /= scaling_factors.unsqueeze(1)

                # When  the analysis is elastic, indexes 1 and 8:end are set to 0
                x[yield_x[:, 0] <= 1 + self.yield_tol, 1] = 0
                x[yield_x[:, 0] <= 1 + self.yield_tol, 8:] = 0

                # In addition, strain energy behaves quadratically, so we divide index 0 another time by the scaling factors, if they are > 1
                x[scaling_factors > 1, 0] /= scaling_factors[scaling_factors > 1]

                # Check whether small-scale plasticity has been violated
                if (yield_x[:, 1] + self.small_scale_pl_tol < yield_x[:, 0]).any():
                    print(
                        "Warning: small-scale plasticity is violated in some analyses. Carefully check results!"
                    )

                # Readjust the sign of the output
                # Tension / Compression
                x[:, 2:6] *= sign_x[:, 0].unsqueeze(1)
                x[:, 8:12] *= sign_x[:, 0].unsqueeze(1)

                # Torsion
                x[:, 6:8] *= sign_x[:, 1].unsqueeze(1)
                x[:, 12:14] *= sign_x[:, 1].unsqueeze(1)

                return x

            else:
                raise ValueError(
                    "YieldNet and SignNet must be defined in production mode"
                )
        else:
            # In training mode, we propagate through the NeuberNet ensemble without any additional processing
            if self.normalize:
                x = (x - self.input_mean) / self.input_std
                x = torch.cat([component(x) for component in self.components], dim=1)
                return x * self.output_std + self.output_mean
            else:
                return torch.cat([component(x) for component in self.components], dim=1)
