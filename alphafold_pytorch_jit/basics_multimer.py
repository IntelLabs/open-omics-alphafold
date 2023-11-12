from torch.nn import functional as F
import torch
from torch import nn
from torch.distributions import Gumbel
from alphafold_pytorch_jit.basics import mask_mean
from alphafold_pytorch_jit import residue_constants
from alphafold.model.geometry import Vec3Array
from alphafold.model.all_atom_multimer import compute_chi_angles
import pdb


def make_msa_profile(
    msa:torch.Tensor,
    msa_mask:torch.Tensor):
  # Compute the profile for every residue (over all MSA sequences).
  return mask_mean(
    msa_mask[:, :, None], 
    F.one_hot(msa, 22), 
    axis=0)


def gumbel_argsort_sample_idx(logits: torch.Tensor) -> torch.Tensor:
  """Samples with replacement from a distribution given by 'logits'.

  This uses Gumbel trick to implement the sampling an efficient manner. For a
  distribution over k items this samples k times without replacement, so this
  is effectively sampling a random permutation with probabilities over the
  permutations derived from the logprobs.

  Args:
    logits: Logarithm of probabilities to sample from, probabilities can be
      unnormalized.

  Returns:
    Sample from logprobs in one-hot form.
  """
  gumbel_src = Gumbel(0., 1.)
  z = gumbel_src.sample(logits.shape)
  perm = torch.argsort(logits + z)
  return torch.flip(perm, dims=[0])


def gumbel_max_sample(logits: torch.Tensor) -> torch.Tensor:
  """Samples from a probability distribution given by 'logits'.
  This uses Gumbel-max trick to implement the sampling in an efficient manner.
  Args:
    key: prng key.
    logits: Logarithm of probabilities to sample from, probabilities can be
      unnormalized.
  Returns:
    Sample from logprobs in one-hot form.
  """
  gumbel_src = Gumbel(0., 1.)
  z = gumbel_src.sample(logits.shape)
  return F.one_hot(
    torch.argmax(logits + z, axis=-1),
    logits.shape[-1]
  ).to(dtype=logits.dtype)


def sample_msa(
    msa:torch.Tensor,
    msa_mask:torch.Tensor,
    max_seq:int,
    cluster_bias_mask:torch.Tensor = None,
    deletion_matrix:torch.Tensor = None,
    bert_mask:torch.Tensor = None):
  """Sample MSA randomly, remaining sequences are stored as `extra_*`.

  Args:
    msa, msa_mask, 
    max_seq: number of sequences to sample,
    cluster_bias_mask
    deletion_matrix,
    bert_mask
  Returns:
    Protein with sampled msa.
  """
  # Sample uniformly among sequences with at least one non-masked position.
  logits = (torch.clip(torch.sum(msa_mask, axis=-1), 0., 1.) - 1.) * 1e6
  # The cluster_bias_mask can be used to preserve the first row (target
  # sequence) for each chain, for example.
  if cluster_bias_mask is None:
    cluster_bias_mask = F.pad(
      torch.zeros(msa.shape[0] - 1), (1, 0), value=1.)

  logits += cluster_bias_mask * 1e6
  index_order = gumbel_argsort_sample_idx(logits)
  sel_idx = index_order[:max_seq]
  extra_idx = index_order[max_seq:]

  # record extra features if needed
  extra_msa = msa[extra_idx]
  msa = msa[sel_idx]
  extra_msa_mask = msa_mask[extra_idx]
  msa_mask = msa_mask[sel_idx]
  if deletion_matrix is not None:
    extra_deletion_matrix = deletion_matrix[extra_idx]
    deletion_matrix = deletion_matrix[sel_idx]
  else:
    extra_deletion_matrix = None
  if bert_mask is not None:
    extra_bert_mask = bert_mask[extra_idx]
    bert_mask = bert_mask[sel_idx]
  else:
    extra_bert_mask = None
  return (
    msa,
    msa_mask,
    cluster_bias_mask,
    extra_msa,
    extra_msa_mask,
    deletion_matrix,
    bert_mask,
    extra_deletion_matrix,
    extra_bert_mask
  )


def make_masked_msa(
  msa:torch.Tensor,
  msa_mask:torch.Tensor,
  msa_profile:torch.Tensor,
  config, 
  bert_mask:torch.Tensor = None,
  epsilon=1e-6,
  dtype=torch.float32
):
  """Create data for BERT on raw MSA."""
  # Add a random amino acid uniformly.
  cfg = config
  random_aa = torch.tensor([0.05] * 20 + [0., 0.], dtype=dtype)
  categorical_probs = (
    cfg['uniform_prob'] * random_aa +
    cfg['profile_prob'] * msa_profile +
    cfg['same_prob'] * F.one_hot(msa, 22))
  # Put all remaining probability on [MASK] which is a new column.
  pad_shapes = [0] * (categorical_probs.dim() * 2)
  pad_shapes[1] = 1 # append to last dimension, last column
  mask_prob = 1. - cfg['profile_prob'] - cfg['same_prob'] - cfg['uniform_prob']
  assert mask_prob >= 0.
  categorical_probs = F.pad(
    categorical_probs, pad_shapes, value=mask_prob) # [TODO] Padding length must be divisible by 2
  uniform = torch.distributions.Uniform(0., 1.)
  mask_position = uniform.sample(msa.shape) < cfg['replace_fraction']
  mask_position = mask_position.to(dtype=dtype)
  mask_position *= msa_mask
  mask_position = mask_position.to(dtype=torch.bool)
  logits = torch.log(categorical_probs + epsilon)
  bert_msa = gumbel_max_sample(logits)
  bert_msa = torch.where(
    mask_position, torch.argmax(bert_msa, dim=-1), msa)
  bert_msa = bert_msa.to(dtype=dtype)
  bert_msa *= msa_mask
  # Mix real and masked MSA.
  if bert_mask is not None:
    bert_mask *= mask_position.to(dtype=dtype)
  else:
    bert_mask = mask_position.to(dtype=dtype)
  true_msa = msa
  msa = bert_msa.to(dtype=torch.int64)
  return (
    msa,
    bert_mask,
    true_msa
  )


def nearest_neighbor_clusters(
  msa,
  msa_mask,
  deletion_matrix,
  extra_msa,
  extra_msa_mask,
  extra_deletion_matrix, 
  gap_agreement_weight=0.
):
  """Assign each extra MSA sequence to its nearest neighbor in sampled MSA."""
  # Determine how much weight we assign to each agreement.  In theory, we could
  # use a full blosum matrix here, but right now let's just down-weight gap
  # agreement because it could be spurious.
  # Never put weight on agreeing on BERT mask.
  weights = torch.tensor(
    [1.] * 21 + [gap_agreement_weight] + [0.], 
    dtype=torch.float32)
  msa_mask = msa_mask
  msa_one_hot = F.one_hot(msa, 23)
  extra_mask = extra_msa_mask
  extra_one_hot = F.one_hot(extra_msa, 23)
  msa_one_hot_masked = msa_mask[:, :, None] * msa_one_hot
  extra_one_hot_masked = extra_mask[:, :, None] * extra_one_hot
  agreement = torch.einsum(
    'mrc, nrc->nm', 
    extra_one_hot_masked,
    weights * msa_one_hot_masked)
  cluster_assignment = F.softmax(1e3 * agreement, dim=0)
  cluster_assignment *= torch.einsum(
    'mr, nr->mn', 
    msa_mask, 
    extra_mask)
  cluster_count = torch.sum(cluster_assignment, dim=-1)
  cluster_count += 1.  # We always include the sequence itself.
  msa_sum = torch.einsum(
    'nm, mrc->nrc', 
    cluster_assignment, 
    extra_one_hot_masked)
  msa_sum += msa_one_hot_masked
  cluster_profile = msa_sum / cluster_count[:, None, None]
  extra_deletion_matrix = extra_deletion_matrix
  deletion_matrix = deletion_matrix
  del_sum = torch.einsum(
    'nm, mc->nc', 
    cluster_assignment,
    extra_mask * extra_deletion_matrix)
  del_sum += deletion_matrix  # Original sequence.
  cluster_deletion_mean = del_sum / cluster_count[:, None]
  return cluster_profile, cluster_deletion_mean


def create_msa_feat(
  msa,
  deletion_matrix,
  cluster_deletion_mean,
  cluster_profile
):
  """Create and concatenate MSA features."""
  msa_1hot = F.one_hot(msa, 23)
  deletion_matrix = deletion_matrix
  has_deletion = torch.clip(deletion_matrix, 0., 1.)[..., None]
  deletion_value = (torch.arctan(deletion_matrix / 3.) * (2. / torch.pi))[..., None]
  deletion_mean_value = (torch.arctan(cluster_deletion_mean / 3.) *
                         (2. / torch.pi))[..., None]
  msa_feat = [
      msa_1hot,
      has_deletion,
      deletion_value,
      cluster_profile,
      deletion_mean_value
  ]
  return torch.concat(msa_feat, dim=-1)


def pseudo_beta_fn(aatype, all_atom_positions, all_atom_masks):
  """Create pseudo beta features."""

  is_gly = torch.tensor([v == residue_constants.restype_order['G'] for v in aatype])
  ca_idx = residue_constants.atom_order['CA']
  cb_idx = residue_constants.atom_order['CB']
  pseudo_beta = torch.where(
      torch.tile(is_gly[..., None], [1] * len(is_gly.shape) + [3]),
      all_atom_positions[..., ca_idx, :],
      all_atom_positions[..., cb_idx, :])

  if all_atom_masks is not None:
    pseudo_beta_mask = torch.where(
        is_gly, all_atom_masks[..., ca_idx], all_atom_masks[..., cb_idx])
    pseudo_beta_mask = pseudo_beta_mask.to(dtype=torch.float32)
    return pseudo_beta, pseudo_beta_mask
  else:
    return pseudo_beta


def create_extra_msa_feature(
  extra_msa,
  extra_msa_mask,
  extra_deletion_matrix,
  num_extra_msa
):
  """Expand extra_msa into 1hot and concat with other extra msa features.

  We do this as late as possible as the one_hot extra msa can be very large.

  Args:
    extra_msa: [num_seq, num_res] MSA that wasn't selected as a cluster
       centre. Note - This isn't one-hotted.
    extra_deletion_matrix: [num_seq, num_res] Number of deletions at given
        position.
    num_extra_msa: Number of extra msa to use.

  Returns:
    Concatenated tensor of extra MSA features.
  """
  # 23 = 20 amino acids + 'X' for unknown + gap + bert mask
  extra_msa = extra_msa[:num_extra_msa]
  deletion_matrix = extra_deletion_matrix[:num_extra_msa]
  msa_1hot = F.one_hot(extra_msa, 23)
  has_deletion = torch.clip(deletion_matrix, 0., 1.)[..., None]
  deletion_value = (torch.arctan(deletion_matrix / 3.) * (2. / torch.pi))[..., None]
  extra_msa_mask = extra_msa_mask[:num_extra_msa]
  return (
    torch.concat([msa_1hot, has_deletion, deletion_value], dim=-1), 
    extra_msa_mask)


class TemplateEmbedding1d(nn.Module): # 源自 modules_multimer.template_embedding_1d
  def __init__(self, global_config, num_channel) -> None:
    super().__init__()
    self.gc = global_config
    # self.num_channel = num_channel
    self.template_single_embedding = nn.Linear(34, num_channel)
    self.template_projection = nn.Linear(num_channel, num_channel)

  def forward(self,
    template_aatype:torch.Tensor, 
    template_all_atom_positions:torch.Tensor,
    template_all_atom_mask:torch.Tensor,
    dtype=torch.float32):
    """Embed templates into an (num_res, num_templates, num_channels) embedding.

    Args:
      template_aatype, (num_templates, num_res) aatype for the templates.
      template_all_atom_positions, (num_templates, num_residues, 37, 3) atom
          positions for the templates.
      template_all_atom_mask, (num_templates, num_residues, 37) atom mask for
          each template.
      num_channel: The number of channels in the output.
      global_config: The global_config.

    Returns:
      An embedding of shape (num_templates, num_res, num_channels) and a mask of
      shape (num_templates, num_res).
    """

    # Embed the templates aatypes.
    aatype_one_hot = F.one_hot(template_aatype, 22)

    num_templates = template_aatype.shape[0]
    all_chi_angles = []
    all_chi_masks = []
    for i in range(num_templates):
      atom_pos = Vec3Array.from_array(
          template_all_atom_positions[i, :, :, :].detach().numpy())
      atom_mask_hk = template_all_atom_mask[i, :, :].detach().numpy()
      t_aatype_hk = template_aatype[i, :].detach().numpy()
      template_chi_angles, template_chi_mask = compute_chi_angles(
          atom_pos,
          atom_mask_hk,
          t_aatype_hk)
      all_chi_angles.append(torch.tensor(template_chi_angles.tolist()))
      all_chi_masks.append(torch.tensor(template_chi_mask.tolist()))
    chi_angles = torch.stack(all_chi_angles, dim=0)
    chi_mask = torch.stack(all_chi_masks, dim=0)
    template_features = torch.concat([
      aatype_one_hot,
      torch.sin(chi_angles) * chi_mask,
      torch.cos(chi_angles) * chi_mask,
      chi_mask], dim=-1)
    template_mask = chi_mask[:, :, 0]
    template_features = template_features.to(dtype=dtype)
    template_mask = template_mask.to(dtype=dtype)
    template_activations = self.template_single_embedding(template_features)
    template_activations = F.relu(template_activations)
    template_activations = self.template_projection(template_activations)
    return template_activations, template_mask
