from typing import Tuple
import torch
from torch import Tensor

__all__ = ["lltm", "reference_lltm"]


def lltm(
    input: Tensor, weights: Tensor, bias: Tensor, old_h: Tensor, old_cell: Tensor
) -> Tuple[Tensor, Tensor]:
    """The lltm API"""
    outputs = torch.ops.extension_cpp.lltm_forward.default(
        input, weights, bias, old_h, old_cell
    )
    new_h, new_cell = outputs[:2]
    return new_h, new_cell


# This is the backward for lltm_forward.
# lltm_forward has 7 returns so they all get gradients.
def backward(ctx, grad_h, grad_cell, _0, _1, _2, _3, _4):
    (
        d_old_h,
        d_input,
        d_weights,
        d_bias,
        d_old_cell,
    ) = torch.ops.extension_cpp.lltm_backward.default(
        grad_h, grad_cell, *ctx.saved_tensors
    )
    return d_input, d_weights, d_bias, d_old_h, d_old_cell


def setup_context(ctx, inputs, output):
    weights = inputs[1]
    new_h, new_cell = output[:2]
    variables = list(output[1:]) + [weights]
    ctx.save_for_backward(*variables)


torch.library.register_autograd(
    "extension_cpp::lltm_forward", backward, setup_context=setup_context)


@torch.library.register_fake("extension_cpp::lltm_forward")
def _(input, weights, bias, old_h, old_cell):
    X = torch.cat([old_h, input], dim=1)
    gate_weights = torch.nn.functional.linear(X, weights, bias)
    gates = gate_weights.chunk(3, dim=1)
    input_gate = torch.empty_like(gates[0])
    output_gate = torch.empty_like(gates[1])
    candidate_cell = torch.empty_like(gates[2])
    new_cell = torch.empty_like(old_cell)
    new_h = torch.empty_like(old_h)
    if input.device.type == "cuda":
        batch_size = old_cell.shape[0]
        state_size = old_cell.shape[1]
        gate_weights = gate_weights.reshape(batch_size, 3, state_size)
    return new_h, new_cell, input_gate, output_gate, candidate_cell, X, gate_weights


def reference_lltm(
    input: Tensor, weights: Tensor, bias: Tensor, old_h: Tensor, old_cell: Tensor
) -> Tuple[Tensor, Tensor]:
    X = torch.cat([old_h, input], dim=1)

    # Compute the input, output and candidate cell gates with one MM.
    gate_weights = torch.nn.functional.linear(X, weights, bias)
    # Split the combined gate weight matrix into its components.
    gates = gate_weights.chunk(3, dim=1)

    input_gate = torch.sigmoid(gates[0])
    output_gate = torch.sigmoid(gates[1])
    # Here we use an ELU instead of the usual tanh.
    candidate_cell = torch.nn.functional.elu(gates[2])

    # Compute the new cell state.
    new_cell = old_cell + candidate_cell * input_gate
    # Compute the new hidden state and output.
    new_h = torch.tanh(new_cell) * output_gate

    return new_h, new_cell
