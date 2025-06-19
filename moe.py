# Sparsely-Gated Mixture-of-Experts Layers.
# See "Outrageously Large Neural Networks"
# https://arxiv.org/abs/1701.06538
#
# Author: David Rau
#
# The code is based on the TensorFlow implementation:
# https://github.com/tensorflow/tensor2tensor/blob/master/tensor2tensor/utils/expert_utils.py


import torch
import torch.nn as nn
from torch.distributions.normal import Normal
import numpy as np


class SparseDispatcher(object):
    """Helper for implementing a mixture of experts.
    The purpose of this class is to create input minibatches for the
    experts and to combine the results of the experts to form a unified
    output tensor.
    There are two functions:
    dispatch - take an input Tensor and create input Tensors for each expert.
    combine - take output Tensors from each expert and form a combined output
      Tensor.  Outputs from different experts for the same batch element are
      summed together, weighted by the provided "gates".
    The class is initialized with a "gates" Tensor, which specifies which
    batch elements go to which experts, and the weights to use when combining
    the outputs.  Batch element b is sent to expert e iff gates[b, e] != 0.
    The inputs and outputs are all two-dimensional [batch, depth].
    Caller is responsible for collapsing additional dimensions prior to
    calling this class and reshaping the output to the original shape.
    See common_layers.reshape_like().
    Example use:
    gates: a float32 `Tensor` with shape `[batch_size, num_experts]`
    inputs: a float32 `Tensor` with shape `[batch_size, input_size]`
    experts: a list of length `num_experts` containing sub-networks.
    dispatcher = SparseDispatcher(num_experts, gates)
    expert_inputs = dispatcher.dispatch(inputs)
    expert_outputs = [experts[i](expert_inputs[i]) for i in range(num_experts)]
    outputs = dispatcher.combine(expert_outputs)
    The preceding code sets the output for a particular example b to:
    output[b] = Sum_i(gates[b, i] * experts[i](inputs[b]))
    This class takes advantage of sparsity in the gate matrix by including in the
    `Tensor`s for expert i only the batch elements for which `gates[b, i] > 0`.
    """

    def __init__(self, num_experts, gates): # gates: (batch_size, expert_score)
        """Create a SparseDispatcher."""

        self._gates = gates
        self._num_experts = num_experts
        # sort experts
        # .nonzero(gates) returns the position of nonzero value in gates 
        # e.g., gates = [[1, 0, 1], [0, 1, 0], [0, 1, 1]], 
        # torch.nonzero(gates) = [[0, 2], [1, 1], [0, 0], [2, 1], [2, 2]], the element's shape means [batch, expert_index] 
        # .sort(0) means order the target by column independently, return the ordered result and original position (indecies) 
        # e.g., [[0, 2], [1, 1], [0, 1]], after sort(0), we have
        # sorted values [[0, 1], [0, 1], [1, 2]], original indeces [[0, 1], [2, 2], [1, 1]]
        # for the example gates, after .nonzero(gates).sort(0), we have 
        # [[0, 0], [0, 1], [1, 1], [2, 2], [2, 2]] and [[0, 2], [2, 1], [1, 3], [3, 0], [4, 4]], which means the original tensor is descending
        sorted_experts, index_sorted_experts = torch.nonzero(gates).sort(0)
        # drop indices
        # get the experts' indecies, but loss the combination between batch and expert
        # _expert_index = [[0], [1], [1], [2], [2]]
        _, self._expert_index = sorted_experts.split(1, dim=1) # indeed, it returns a tuple
        # get according batch index for each expert
        # index_sorted_experts[:, 1] = [2, 1, 3, 0, 4], the original experts' indecies 
        # torch.nonzero(gates)[index_sorted_experts[:, 1], 0] = [0, 1, 2, 0, 2]
        self._batch_index = torch.nonzero(gates)[index_sorted_experts[:, 1], 0] # the corresponding batch of _expert_index
        # calculate num samples that each expert gets
        self._part_sizes = (gates > 0).sum(0).tolist()
        # expand gates to match with self._batch_index
        # [[1, 0, 1], [0, 1, 0], [0, 1, 1]]
        # gates_exp = [[1, 0, 1], [0, 1, 0], [0, 1, 1], [1, 0, 1], [0, 1, 1]]
        # gates_exp is the batches for each expert from expert_0 to expert_e
        gates_exp = gates[self._batch_index.flatten()] 
        # [1, 1, 1, 1, 1]
        # _nonzero_gates is the weights for each expert in the corresponding batch
        self._nonzero_gates = torch.gather(gates_exp, 1, self._expert_index)

    def dispatch(self, inp):
        """Create one input Tensor for each expert.
        The `Tensor` for a expert `i` contains the slices of `inp` corresponding
        to the batch elements `b` where `gates[b, i] > 0`.
        Args:
          inp: a `Tensor` of shape "[batch_size, <extra_input_dims>]`
        Returns:
          a list of `num_experts` `Tensor`s with shapes
            `[expert_batch_size_i, <extra_input_dims>]`.
        """

        # assigns samples to experts whose gate is nonzero

        # expand according to batch index so we can just split by _part_sizes
        inp_exp = inp[self._batch_index].squeeze(1) # squeeze ensures the dimension is [batch_size, features], not like [batch_size, 1, features]
        return torch.split(inp_exp, self._part_sizes, dim=0)

    def combine(self, expert_out, multiply_by_gates=True):
        """Sum together the expert output, weighted by the gates.
        The slice corresponding to a particular batch element `b` is computed
        as the sum over all experts `i` of the expert output, weighted by the
        corresponding gate values.  If `multiply_by_gates` is set to False, the
        gate values are ignored.
        Args:
          expert_out: a list of `num_experts` `Tensor`s, each with shape
            `[expert_batch_size_i, <extra_output_dims>]`.
          multiply_by_gates: a boolean
        Returns:
          a `Tensor` with shape `[batch_size, <extra_output_dims>]`.
        """
        # apply exp to expert outputs, so we are not longer in log space
        stitched = torch.cat(expert_out, 0)

        if multiply_by_gates:
            stitched = stitched.mul(self._nonzero_gates) # _nonzero_gates will be broadcast
        zeros = torch.zeros(self._gates.size(0), expert_out[-1].size(1), requires_grad=True, device=stitched.device) # shape: (batch_size, <extra_output_dims>)
        # combine samples that have been processed by the same k experts
        combined = zeros.index_add(0, self._batch_index, stitched.float())
        return combined

    def expert_to_gates(self):
        """Gate values corresponding to the examples in the per-expert `Tensor`s.
        Returns:
          a list of `num_experts` one-dimensional `Tensor`s with type `tf.float32`
              and shapes `[expert_batch_size_i]`
        """
        # split nonzero gates for each expert
        return torch.split(self._nonzero_gates, self._part_sizes, dim=0)

class MLP(nn.Module):
    def __init__(self, input_size, output_size, hidden_size):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, output_size)
        self.relu = nn.ReLU()
        self.soft = nn.Softmax(1)

    def forward(self, x):
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        out = self.soft(out)
        return out


class MoE(nn.Module):

    """Call a Sparsely gated mixture of experts layer with 1-layer Feed-Forward networks as experts.
    Args:
    input_size: integer - size of the input
    output_size: integer - size of the input
    num_experts: an integer - number of experts
    hidden_size: an integer - hidden size of the experts
    noisy_gating: a boolean
    k: an integer - how many experts to use for each batch element
    """

    def __init__(self, input_size, output_size, num_experts, hidden_size, noisy_gating=True, k=4):
        super(MoE, self).__init__()
        self.noisy_gating = noisy_gating
        self.num_experts = num_experts
        self.output_size = output_size
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.k = k
        # instantiate experts
        self.experts = nn.ModuleList([MLP(self.input_size, self.output_size, self.hidden_size) for i in range(self.num_experts)])
        self.w_gate = nn.Parameter(torch.zeros(input_size, num_experts), requires_grad=True)
        self.w_noise = nn.Parameter(torch.zeros(input_size, num_experts), requires_grad=True)

        self.softplus = nn.Softplus()
        self.softmax = nn.Softmax(1)
        self.register_buffer("mean", torch.tensor([0.0]))
        self.register_buffer("std", torch.tensor([1.0]))
        assert(self.k <= self.num_experts)

    def cv_squared(self, x): # Coefficient of Variation, CV
        """The squared coefficient of variation of a sample.
        Useful as a loss to encourage a positive distribution to be more uniform.
        Epsilons added for numerical stability.
        Returns 0 for an empty Tensor.
        Args:
        x: a `Tensor`.
        Returns:
        a `Scalar`.
        """
        eps = 1e-10
        # if only num_experts = 1

        if x.shape[0] == 1:
            return torch.tensor([0], device=x.device, dtype=x.dtype)
        return x.float().var() / (x.float().mean()**2 + eps)

    def _gates_to_load(self, gates):
        """Compute the true load per expert, given the gates.
        The load is the number of examples for which the corresponding gate is >0.
        Args:
        gates: a `Tensor` of shape [batch_size, n]
        Returns:
        a float32 `Tensor` of shape [n]
        """
        return (gates > 0).sum(0)

    def _prob_in_top_k(self, clean_values, noisy_values, noise_stddev, noisy_top_values):
        """Helper function to NoisyTopKGating.
        Computes the probability that value is in top k, given different random noise.
        This gives us a way of backpropagating from a loss that balances the number
        of times each expert is in the top k experts per example.
        In the case of no noise, pass in None for noise_stddev, and the result will
        not be differentiable.
        Args:
        clean_values: a `Tensor` of shape [batch, n].
        noisy_values: a `Tensor` of shape [batch, n].  Equal to clean values plus
          normally distributed noise with standard deviation noise_stddev.
        noise_stddev: a `Tensor` of shape [batch, n], or None
        noisy_top_values: a `Tensor` of shape [batch, m].
           "values" Output of tf.top_k(noisy_top_values, m).  m >= k+1
        Returns:
        a `Tensor` of shape [batch, n].
        """
        # clean_values: [batch_size, num_experts] (即 clean_logits)
        # noisy_values: [batch_size, num_experts] (即 noisy_logits)
        # noise_stddev: [batch_size, num_experts] (每个专家对应的噪声标准差)
        # noisy_top_values: [batch_size, min(k+1, num_experts)] (由 noisy_logits.topk 得到的前 k+1 个最大值)

        batch = clean_values.size(0) # batch_size
        m = noisy_top_values.size(1) # min(self.k + 1, self.num_experts)

        # 1. 展平 noisy_top_values
        # noisy_top_values: [batch_size, m]
        # top_values_flat: [batch_size * m]
        top_values_flat = noisy_top_values.flatten()

        # 2. 计算 threshold_if_in (第 k 大的 noisy logit 作为阈值)
        # threshold_positions_if_in: [batch_size]
        # (例如，如果 batch_size=2, m=3, k=1, 那么 torch.arange(2)*3 + 1 = [1, 4]
        # 这表示对于第一个样本，我们取 flat 后的第1个元素 (top_values_flat[1])，即它第二大的值
        # 对于第二个样本，取 flat 后的第4个元素 (top_values_flat[4])，即它第二大的值
        # 这里的 k 是 0-indexed，所以 k 表示第 (k+1) 大的值
        # 如果 self.k = 1, 则取第 2 大的值作为阈值
        # 如果 self.k = 0, 则取第 1 大的值作为阈值 (即最大值)
        # 这个索引指向每个样本的第 k 个 top 值 (0-indexed)
        threshold_positions_if_in = torch.arange(batch, device=clean_values.device) * m + self.k
        # torch.gather(input, dim, index) 从 input 中沿着 dim 收集 index 指定的元素
        # 这里 input 是 flat 的 top_values_flat，dim=0，index 是 threshold_positions_if_in
        # threshold_if_in: [batch_size] (然后通过 unsqueeze 变为 [batch_size, 1])
        threshold_if_in = torch.unsqueeze(torch.gather(top_values_flat, 0, threshold_positions_if_in), 1)
        # threshold_if_in 形状: [batch_size, 1]
        # 含义: 对于每个样本，其第 k 个 (0-indexed) 最大的带噪声门控分数。
        # 如果一个专家其 clean_value + noise 大于这个阈值，它就在 top k 之内。

        # 3. 判断哪些专家在 noisy top k 内 (实际发生了什么)
        # is_in: [batch_size, num_experts] (布尔张量)
        is_in = torch.gt(noisy_values, threshold_if_in)
        # torch.gt(a, b) 返回一个布尔张量，表示 a > b。
        # 这里 noisy_values 形状是 [batch_size, num_experts]，
        # threshold_if_in 形状是 [batch_size, 1]，会自动广播到 [batch_size, num_experts]。
        # 结果是一个布尔掩码，指示在应用噪声后，哪些专家实际进入了 Top-K。

        # 4. 计算 threshold_if_out (第 k-1 大的 noisy logit 作为阈值)
        # threshold_positions_if_out: [batch_size]
        # 这里的索引指向每个样本的第 k-1 个 top 值 (0-indexed)，即 Top-K 之外的最高值。
        threshold_positions_if_out = threshold_positions_if_in - 1
        # threshold_if_out: [batch_size, 1]
        threshold_if_out = torch.unsqueeze(torch.gather(top_values_flat, 0, threshold_positions_if_out), 1)
        # 含义: 对于每个样本，其第 k-1 个 (0-indexed) 最大的带噪声门控分数。
        # 这个阈值用于计算当专家实际不在 Top-K 中时，它进入 Top-K 的概率。

        # 5. 计算概率 (Probabilities using CDF)
        # self.mean: 0.0 (from register_buffer)
        # self.std: 1.0 (from register_buffer)
        normal = Normal(self.mean, self.std) # 标准正态分布

        # prob_if_in: [batch_size, num_experts]
        # 计算每个 clean_value + N(0, noise_stddev) > threshold_if_in 的概率
        prob_if_in = normal.cdf((clean_values - threshold_if_in)/noise_stddev)
        # prob_if_out: [batch_size, num_experts]
        # 计算每个 clean_value + N(0, noise_stddev) > threshold_if_out 的概率
        prob_if_out = normal.cdf((clean_values - threshold_if_out)/noise_stddev)
        # 注意：这里的 CDF 计算的是 P(Z < X)，而我们想要的是 P(Z > X) = 1 - P(Z < X)
        # 或者 P(clean_value + noise > threshold) => P(noise > threshold - clean_value)
        # Z = noise / noise_stddev ~ N(0,1)
        # P(Z > (threshold - clean_value) / noise_stddev) = 1 - CDF((threshold - clean_value) / noise_stddev)
        # 由于正态分布对称性：1 - CDF(x) = CDF(-x)
        # 因此，1 - CDF((threshold - clean_value) / noise_stddev) = CDF(-(threshold - clean_value) / noise_stddev) = CDF((clean_value - threshold) / noise_stddev)
        # 这就是代码中直接使用 (clean_values - threshold) / noise_stddev 作为输入的原因。

        # 6. 根据实际 Top-K 结果选择合适的概率
        # prob: [batch_size, num_experts]
        prob = torch.where(is_in, prob_if_in, prob_if_out)
        # 对于每个专家，如果它在 noisy_values 中实际进入了 top k (is_in 为 True)，
        # 则使用 prob_if_in (即 clean_value 超过 top k 阈值的概率)。
        # 如果它没有进入 top k (is_in 为 False)，
        # 则使用 prob_if_out (即 clean_value 超过 top k-1 阈值的概率)。
        # 这种选择是为了在梯度下降时，对于那些“刚好”在 Top-K 边缘的专家，提供更平滑的梯度信号，
        # 促使它们的 clean_logits 要么明确地进入 Top-K，要么明确地离开 Top-K，从而平衡负载。

        return prob

    def noisy_top_k_gating(self, x, train, noise_epsilon=1e-2):
        """Noisy top-k gating.
          See paper: https://arxiv.org/abs/1701.06538.
          Args:
            x: input Tensor with shape [batch_size, input_size]
            train: a boolean - we only add noise at training time.
            noise_epsilon: a float
          Returns:
            gates: a Tensor with shape [batch_size, num_experts]
            load: a Tensor with shape [num_experts]
        """
        # x: [batch_size, input_size]

        # 1. 计算专家选择的“干净”分数 (Clean Logits)
        # self.w_gate: [input_size, num_experts]
        # clean_logits: [batch_size, input_size] @ [input_size, num_experts] = [batch_size, num_experts]
        clean_logits = x @ self.w_gate

        # 2. 有条件地添加噪声 (Noisy Gating)
        if self.noisy_gating and train:
            # self.w_noise: [input_size, num_experts]
            # raw_noise_stddev: [batch_size, input_size] @ [input_size, num_experts] = [batch_size, num_experts]
            raw_noise_stddev = x @ self.w_noise
            # softplus(x) = log(1 + exp(x)), 确保标准差非负且可微分
            # noise_stddev: [batch_size, num_experts]
            noise_stddev = (self.softplus(raw_noise_stddev) + noise_epsilon)
            # torch.randn_like(clean_logits) 生成与 clean_logits 形状相同的标准正态随机数
            # noisy_logits = clean_logits + 噪声 (每个元素独立采样)
            # noisy_logits: [batch_size, num_experts]
            noisy_logits = clean_logits + (torch.randn_like(clean_logits) * noise_stddev)
            logits = noisy_logits
        else:
            # 评估模式或禁用噪声时，直接使用干净分数
            logits = clean_logits
        # 此时 logits 形状为 [batch_size, num_experts]

        # 3. 将 logits 转换为概率分布 (Softmax)
        # logits: [batch_size, num_experts]
        logits = self.softmax(logits) # 对每个样本，其所有专家的分数和为 1

        # 4. 选择 Top-K 专家及其门控权重
        # topk(input, k, dim) 返回值和索引
        # min(self.k + 1, self.num_experts) 是为了在计算 _prob_in_top_k 时提供足够的阈值信息
        # top_logits: [batch_size, min(k+1, num_experts)]
        # top_indices: [batch_size, min(k+1, num_experts)]
        top_logits, top_indices = logits.topk(min(self.k + 1, self.num_experts), dim=1)

        # 选取实际的 Top-K 部分
        # top_k_logits: [batch_size, k]
        top_k_logits = top_logits[:, :self.k]
        # top_k_indices: [batch_size, k]
        top_k_indices = top_indices[:, :self.k]

        # 归一化 Top-K 门控权重，使每个样本选中的 k 个专家权重之和为 1
        # top_k_gates: [batch_size, k]
        top_k_gates = top_k_logits / (top_k_logits.sum(1, keepdim=True) + 1e-6)  # 1e-6 避免除以零

        # 5. 构建稀疏的门控张量 (Full Gates Tensor)
        # zeros: [batch_size, num_experts]，全零张量
        zeros = torch.zeros_like(logits, requires_grad=True)
        # scatter(dim, index, src) 将 src 中的值，根据 index 散布到 dim 指定的维度上
        # gates: [batch_size, num_experts]
        gates = zeros.scatter(1, top_k_indices, top_k_gates)
        # 此时，gates 张量中只有每个样本被选中的 k 个专家对应的位置有非零值 (归一化后的权重)，其他为零。

        # 6. 计算专家负载 (Load)
        if self.noisy_gating and self.k < self.num_experts and train:
            # 如果使用噪声门控且在训练模式，且并非所有专家都选中，则使用可微分的概率来计算负载
            # load: [num_experts]
            load = (self._prob_in_top_k(clean_logits, noisy_logits, noise_stddev, top_logits)).sum(0)
        else:
            # 否则，使用简单的非可微分计数方式计算负载
            # load: [num_experts]
            load = self._gates_to_load(gates) # 统计 gates > 0 的数量

        return gates, load

    def forward(self, x, loss_coef=1e-2):
        """Args:
        x: tensor shape [batch_size, input_size]
        train: a boolean scalar.
        loss_coef: a scalar - multiplier on load-balancing losses

        Returns:
        y: a tensor with shape [batch_size, output_size].
        extra_training_loss: a scalar.  This should be added into the overall
        training loss of the model.  The backpropagation of this loss
        encourages all experts to be approximately equally used across a batch.
        """
        gates, load = self.noisy_top_k_gating(x, self.training)
        # calculate importance loss
        importance = gates.sum(0)
        #
        loss = self.cv_squared(importance) + self.cv_squared(load)
        loss *= loss_coef

        dispatcher = SparseDispatcher(self.num_experts, gates)
        expert_inputs = dispatcher.dispatch(x)
        gates = dispatcher.expert_to_gates()
        expert_outputs = [self.experts[i](expert_inputs[i]) for i in range(self.num_experts)]
        y = dispatcher.combine(expert_outputs)
        return y, loss
