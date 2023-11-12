import torch
from torch import nn


class FusedTriangleMultiplication(nn.Module):

  def __init__(self,config, global_config, act_dim):
    """Builds TriangleMultiplication module, w/ fused projection weights
    Arguments:
      act: Pair activations, shape [N_res, N_res, c_z]
      mask: Pair mask, shape [N_res, N_res].
      is_training: Whether the module is in training mode.
    Returns:
      Outputs, same shape/type as act.
    """
    super().__init__()
    self.c = config
    self.gc = global_config
    self.c_equation = self.c['equation']
    self.left_norm_input = nn.LayerNorm(act_dim)
    self.num_intermediate_channel = self.c['num_intermediate_channel']
    self.projection = nn.Linear(
      act_dim, 
      2* self.num_intermediate_channel)
    self.gate = nn.Linear(
      2* self.num_intermediate_channel,
      2* self.num_intermediate_channel)
    self.center_norm = nn.LayerNorm(self.num_intermediate_channel)
    self.output_projection = nn.Linear(self.num_intermediate_channel,act_dim)
    self.gating_linear = nn.Linear(act_dim,act_dim)

  def forward(self, act, mask):
    mask = mask[..., None]
    left_act = self.left_norm_input(act)
    proj_act = mask * self.projection(left_act)
    proj_act *= torch.sigmoid(self.gate(left_act))
    left_proj_act = proj_act[:, :, :self.num_intermediate_channel]
    right_proj_act = proj_act[:, :, self.num_intermediate_channel:]
    act = torch.einsum(self.c_equation, left_proj_act, right_proj_act)
    act = self.center_norm(act)
    act = self.output_projection(act)
    gated_value = self.gating_linear(act)
    ori_value = act
    act = torch.sigmoid(gated_value) * ori_value
    return act
