import jax
import haiku as hk


def get_pure_fn(model, c, gc):
  """
  get pure function
  return init,apply
  """
  def _forward(*args):
    mod = model(c,gc)
    return mod(*args)

  init = jax.jit(hk.transform(_forward).init)
  apply = jax.jit(hk.transform(_forward).apply)
  return init,apply