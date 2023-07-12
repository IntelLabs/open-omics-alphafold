
### We can accelerate the attention module in intel-alphafold2 codebase by replacing it with the optimized version from [tpp_pytorch_extension].
### The following lines of code in intel-alphafold2 codebase (https://github.com/IntelAI/models/tree/master/models/aidd/pytorch/alphafold2/) 
### replaces the existing forward pass of the GatingAttention module with the optimized version.
### Add the following lines of code in inference/run_modelinfer_pytorch_jit.py at the top of the file.

```python
from alphafold_pytorch_jit.basics import GatingAttention
from tpp_pytorch_extension.alphafold.Alpha_Attention import GatingAttentionOpti_forward
GatingAttention.forward = GatingAttentionOpti_forward
```

### For unit testing and comparision purpose, we can also use the optimized attention layer (GatingAttentionOpti) from [tpp_pytorch_extension] as follows:

```python
from tpp_pytorch_extension.alphafold.Alpha_Attention import GatingAttentionOpti

class Net2(nn.Module):  # Network containing optimized attention layer
    def __init__(self):
        super(Net2, self).__init__()
        self.attention = GatingAttentionOpti(
            num_head=N, a_dim=HS, m_dim=HS, output_dim=HS
        )  # Attention layer

    def forward(self, q_data, m_data, bias, nonbatched_bias):
        x = self.attention(q_data, m_data, bias, nonbatched_bias)
        return x
```

### attention_examaple.py and attention_example_bf16.py files in this folder contain unit tests to compare 
### the original GatingAttention layer with the optimized attention layer (GatingAttentionOpti).