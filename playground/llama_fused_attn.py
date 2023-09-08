import time
import torch
import transformers
from torch._inductor.utils import run_and_get_code, print_performance
import math
from torch.profiler import profile, record_function, ProfilerActivity
from torch._dynamo.utils import counters

# works
def model0(query, key, value):
    q = query.permute(0, 2, 1, 3)
    k = key.permute(0, 2, 1, 3)
    v = value.permute(0, 2, 1, 3)
    div = q @ k.transpose(-2, -1) / math.sqrt(q.size(-1))
    div = div.to(torch.float32)
    attn_weight = torch.softmax(div, dim=-1)
    # very small dropout to make sure test passes
    # attn_weight = torch.dropout(attn_weight, 0.00001, True)
    attn_weight = attn_weight.to(torch.float16)
    return attn_weight @ v

# works
def model1(query, key, value):
    attn_weights = torch.matmul(query, key.transpose(2, 3)) / math.sqrt(query.size(-1))
    # attn_weights = torch.nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query.dtype)
    attn_weights = torch.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query.dtype)
    attn_output = torch.matmul(attn_weights, value)
    return attn_output
device = torch.device('cuda:0')

args0 = (
    torch.randn((2, 4, 8, 16), device=device, dtype=torch.half),
    torch.randn((2, 4, 8, 16), device=device, dtype=torch.half),
    torch.randn((2, 4, 8, 16), device=device, dtype=torch.half),
)

args1 = (
    torch.randn((2, 8, 4, 16), device=device, dtype=torch.half),
    torch.randn((2, 8, 4, 16), device=device, dtype=torch.half),
    torch.randn((2, 8, 4, 16), device=device, dtype=torch.half),
)

model = model1
args = args1
kwargs = {k: v for k,v in zip(['query','key','value'], args)}

with torch.no_grad():
    # tmp = model(ins)
    result2, (source_code,) = run_and_get_code(
        torch.compile(model), **kwargs
    )
    with open("fused_code.py", 'w') as fh:
        fh.write(source_code)

print(counters["fuse_attention"])
print(torch.testing.assert_allclose(model(*args),result2))
