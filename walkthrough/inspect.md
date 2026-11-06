# Inspect one tensor

Click a row in the tensor table. The panel on the right shows:

- min, max, mean, std, and the NaN, +Inf, -Inf and zero counts
- a heatmap for 2-D tensors, a line plot for 1-D tensors, or the picture for an HxWx3 uint8 array
- a histogram of the values

GGUF tensors in F32, F16, BF16, Q8_0, Q4_0, Q4_1, Q5_0, Q5_1, Q4_K, Q5_K and Q6_K are decoded. Other quant types are listed with their size.

Copy shape gives a Python tuple like `(4096, 11008)`. Copy torch.Size gives `torch.Size([4096, 11008])`.
