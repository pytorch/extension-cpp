#include <torch/extension.h>

#include <vector>

// s'(z) = (1 - s(z)) * s(z)
torch::Tensor d_sigmoid(torch::Tensor z) {
  auto s = torch::sigmoid(z);
  return (1 - s) * s;
}

// tanh'(z) = 1 - tanh^2(z)
torch::Tensor d_tanh(torch::Tensor z) {
  return 1 - z.tanh().pow(2);
}

// elu'(z) = relu'(z) + { alpha * exp(z) if (alpha * (exp(z) - 1)) < 0, else 0}
torch::Tensor d_elu(torch::Tensor z, torch::Scalar alpha = 1.0) {
  auto e = z.exp();
  auto mask = (alpha * (e - 1)) < 0;
  return (z > 0).type_as(z) + mask.type_as(z) * (alpha * e);
}

std::vector<torch::Tensor> lltm_forward(
    torch::Tensor input,
    torch::Tensor weights,
    torch::Tensor bias,
    torch::Tensor old_h,
    torch::Tensor old_cell) {

  auto X = torch::cat({old_h, input}, /*dim=*/1);

  auto gate_weights = torch::addmm(bias, X, weights.transpose(0, 1));
  auto gates = gate_weights.chunk(3, /*dim=*/1);

  auto input_gate = torch::sigmoid(gates[0]);
  auto output_gate = torch::sigmoid(gates[1]);
  auto candidate_cell = torch::elu(gates[2], /*alpha=*/1.0);

  auto new_cell = old_cell + candidate_cell * input_gate;
  auto new_h = torch::tanh(new_cell) * output_gate;

  return {new_h,
          new_cell,
          input_gate,
          output_gate,
          candidate_cell,
          X,
          gate_weights};
}

std::vector<torch::Tensor> lltm_backward(
    torch::Tensor grad_h,
    torch::Tensor grad_cell,
    torch::Tensor new_cell,
    torch::Tensor input_gate,
    torch::Tensor output_gate,
    torch::Tensor candidate_cell,
    torch::Tensor X,
    torch::Tensor gate_weights,
    torch::Tensor weights) {
  auto d_output_gate = torch::tanh(new_cell) * grad_h;
  auto d_tanh_new_cell = output_gate * grad_h;
  auto d_new_cell = d_tanh(new_cell) * d_tanh_new_cell + grad_cell;

  auto d_old_cell = d_new_cell;
  auto d_candidate_cell = input_gate * d_new_cell;
  auto d_input_gate = candidate_cell * d_new_cell;

  auto gates = gate_weights.chunk(3, /*dim=*/1);
  d_input_gate *= d_sigmoid(gates[0]);
  d_output_gate *= d_sigmoid(gates[1]);
  d_candidate_cell *= d_elu(gates[2]);

  auto d_gates =
      torch::cat({d_input_gate, d_output_gate, d_candidate_cell}, /*dim=*/1);

  auto d_weights = d_gates.t().mm(X);
  auto d_bias = d_gates.sum(/*dim=*/0, /*keepdim=*/true);

  auto d_X = d_gates.mm(weights);
  const auto state_size = grad_h.size(1);
  auto d_old_h = d_X.slice(/*dim=*/1, 0, state_size);
  auto d_input = d_X.slice(/*dim=*/1, state_size);

  return {d_old_h, d_input, d_weights, d_bias, d_old_cell};
}

std::vector<torch::Tensor> lltm_op(torch::Tensor input,
				   torch::Tensor weights,
				   torch::Tensor bias,
				   torch::Tensor old_h,
				   torch::Tensor old_cell){
  static auto op = torch::Dispatcher::singleton()
    .findSchemaOrThrow("myops::lltm", "")
    .typed<decltype(lltm_op)>();
  return op.call(input, weights, bias, old_h, old_cell);
}

std::vector<torch::Tensor> lltm_op_backward(torch::Tensor grad_h,
					    torch::Tensor grad_cell,
					    torch::Tensor new_cell,
					    torch::Tensor input_gate,
					    torch::Tensor output_gate,
					    torch::Tensor candidate_cell,
					    torch::Tensor X,
					    torch::Tensor gate_weights,
					    torch::Tensor weights){
  static auto op = torch::Dispatcher::singleton()
    .findSchemaOrThrow("myops::lltm", "backward")
    .typed<decltype(lltm_op_backward)>();
  return op.call(grad_h, grad_cell, new_cell, input_gate,
		 output_gate, candidate_cell, X, gate_weights, weights);
}

class LLTMFunction : public torch::autograd::Function<LLTMFunction> {
public:
  static std::vector<torch::Tensor> forward(torch::autograd::AutogradContext *ctx,
					    torch::Tensor input,
					    torch::Tensor weights,
					    torch::Tensor bias,
					    torch::Tensor old_h,
					    torch::Tensor old_cell){
    at::AutoDispatchBelowADInplaceOrView g;
    std::vector<torch::Tensor> outputs = lltm_op(input, weights, bias, old_h, old_cell);
    ctx->save_for_backward({outputs[1], outputs[2], outputs[3],
	outputs[4], outputs[5], outputs[6], weights});
        
    return {outputs[0], outputs[1]};
  }

  static torch::autograd::tensor_list backward(torch::autograd::AutogradContext *ctx,
					       torch::autograd::tensor_list grad_outputs){
    auto saved = ctx->get_saved_variables();
    auto outputs = lltm_op_backward(grad_outputs[0].contiguous(),
				    grad_outputs[1].contiguous(),
				    saved[0], saved[1], saved[2], saved[3],
				    saved[4], saved[5], saved[6]);
    return {outputs[1], outputs[2], outputs[3], outputs[0], outputs[4]};
  }
};

std::vector<torch::Tensor> lltm_autograd(torch::Tensor input,
					 torch::Tensor weights,
					 torch::Tensor bias,
					 torch::Tensor old_h,
					 torch::Tensor old_cell) {
  return LLTMFunction::apply(input, weights, bias, old_h, old_cell);
}

TORCH_LIBRARY(myops, m){
  m.def("lltm(Tensor input, Tensor weights, Tensor bias, Tensor old_h, Tensor old_cell)" \
	"-> Tensor[]");
  m.def("lltm.backward(Tensor grad_h, Tensor grad_cell, Tensor new_cell, " \
	"Tensor input_gate, Tensor output_gate, Tensor candidate_cell, Tensor X, " \
	"Tensor gate_weights, Tensor weights) -> Tensor[]");
}

TORCH_LIBRARY_IMPL(myops, CPU, m){
  m.impl(TORCH_SELECTIVE_NAME("lltm"), TORCH_FN(lltm_forward));
  m.impl(TORCH_SELECTIVE_NAME("lltm.backward"), TORCH_FN(lltm_backward));
}

TORCH_LIBRARY_IMPL(myops, Autograd, m) {
  m.impl("lltm", lltm_autograd);
}
