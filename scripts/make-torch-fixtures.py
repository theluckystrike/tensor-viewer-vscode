"""Real torch.save files for test/torch.test.js. usage: python3 scripts/make-torch-fixtures.py test/fixtures/torch [--big]
Writes expected.json with torch's own view of every tensor (shape, dtype, sum, first values)."""
import sys, os, io, zipfile, pickle, collections, json
import torch, torch.nn as nn
out = sys.argv[1]; os.makedirs(out, exist_ok=True)
torch.manual_seed(0)
P = lambda n: os.path.join(out, n)
exp = {}

class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 8, 3); self.bn1 = nn.BatchNorm2d(8)
        self.fc = nn.Linear(8 * 6 * 6, 10)
    def forward(self, x): return self.fc(torch.flatten(self.bn1(self.conv1(x)), 1))
net = Net()
sd = net.state_dict()
torch.save(sd, P('state_dict.pt'))
exp['state_dict.pt'] = {k: [list(v.shape), str(v.dtype)] for k, v in sd.items()}

opt = torch.optim.Adam(net.parameters(), lr=1e-3)
net(torch.randn(2, 3, 8, 8)).sum().backward(); opt.step()
ck = {'epoch': 7, 'model_state_dict': net.state_dict(), 'optimizer_state_dict': opt.state_dict(), 'loss': 0.4321, 'note': 'run-a'}
torch.save(ck, P('checkpoint.pth'))

# Lightning-style .ckpt
torch.save({'epoch': 3, 'global_step': 1200, 'pytorch-lightning_version': '2.4.0', 'state_dict': {'model.' + k: v for k, v in net.state_dict().items()},
            'hyper_parameters': {'lr': 0.001, 'hidden': 64, 'layers': [1, 2, 3]}, 'callbacks': {}, 'lr_schedulers': []}, P('lightning.ckpt'))

torch.save(torch.arange(12, dtype=torch.float32).reshape(3, 4), P('single_tensor.pt'))

d = collections.OrderedDict()
d['f16'] = torch.randn(4, 5).half(); d['bf16'] = torch.randn(4, 5).bfloat16(); d['f64'] = torch.randn(3, 3, dtype=torch.float64)
d['i64'] = torch.arange(-5, 5); d['i32'] = torch.arange(10, dtype=torch.int32); d['i16'] = torch.arange(6, dtype=torch.int16)
d['i8'] = torch.arange(-3, 3, dtype=torch.int8); d['u8'] = torch.arange(0, 250, 10, dtype=torch.uint8); d['bool'] = torch.tensor([True, False, True])
d['c64'] = torch.complex(torch.ones(3), torch.ones(3)); d['scalar'] = torch.tensor(3.5); d['empty'] = torch.zeros(0, 4)
d['f8e4m3'] = torch.tensor([0.5, 1.0, -2.0]).to(torch.float8_e4m3fn); d['f8e5m2'] = torch.tensor([0.5, 1.0, -2.0]).to(torch.float8_e5m2)
torch.save(d, P('dtypes.pt'))
exp['dtypes.pt'] = {k: [list(v.shape), str(v.dtype), float(v.float().sum()) if v.numel() and not v.is_complex() else None] for k, v in d.items()}

base = torch.arange(24, dtype=torch.float32).reshape(4, 6)
views = {'base': base, 'transposed': base.t(), 'row_slice': base[2:], 'col_slice': base[:, 1:3], 'shared_flat': base.view(-1)[5:11]}
torch.save(views, P('views.pt'))
exp['views.pt'] = {k: [list(v.shape), float(v.sum()), v.flatten()[:4].tolist()] for k, v in views.items()}

torch.save(sd, P('legacy.pth'), _use_new_zipfile_serialization=False)
torch.save(net, P('whole_module.pt'))
torch.jit.save(torch.jit.script(net), P('scripted.pt'))
torch.save(sd, P('pytorch_model.bin'))
q = torch.quantize_per_tensor(torch.randn(4, 4), 0.1, 0, torch.quint8)
torch.save({'q': q, 'sparse': torch.eye(3).to_sparse(), 'dense': torch.ones(2, 2)}, P('quant_sparse.pt'))

# A hostile .pt: data.pkl calls posix.system. Built by hand; nothing here runs it.
class Evil:
    def __reduce__(self): return (os.system, ('echo pwned > /tmp/tens2-pwned',))
buf = io.BytesIO(); pickle.dump({'weight': 1, 'x': Evil()}, buf, protocol=2)
with zipfile.ZipFile(P('evil.pt'), 'w', zipfile.ZIP_STORED) as z:
    z.writestr('evil/data.pkl', buf.getvalue()); z.writestr('evil/version', '3\n')
raw = open(P('state_dict.pt'), 'rb').read()
open(P('truncated.pt'), 'wb').write(raw[: len(raw) // 2])
if '--big' in sys.argv:
    big = collections.OrderedDict((f'layers.{i}.weight', torch.randn(1024, 1024)) for i in range(12))
    big['embed.weight'] = torch.randn(8192, 512).half()
    torch.save(big, P('big_model.pt'))

def flat(v, path, out):
    if isinstance(v, torch.Tensor):
        if v.is_sparse or v.is_quantized: out[path or '(tensor)'] = {'shape': list(v.shape), 'dtype': str(v.dtype).replace('torch.', '')}; return
        f = v.detach().to(torch.float64) if not v.is_complex() else v.detach().abs().to(torch.float64)
        out[path or '(tensor)'] = {'shape': list(v.shape), 'dtype': str(v.dtype).replace('torch.', ''), 'sum': float(f.sum()) if v.numel() else 0.0,
                                   'first': f.flatten()[:4].tolist()}
    elif isinstance(v, dict):
        for k, x in v.items(): flat(x, f'{path}.{k}' if path else str(k), out)
    elif isinstance(v, (list, tuple)):
        for i, x in enumerate(v): flat(x, f'{path}.{i}' if path else str(i), out)
full = {}
for f in ['state_dict.pt', 'checkpoint.pth', 'lightning.ckpt', 'single_tensor.pt', 'dtypes.pt', 'views.pt', 'legacy.pth', 'pytorch_model.bin', 'quant_sparse.pt']:
    o = {}; flat(torch.load(P(f), weights_only=True), '', o); full[f] = o
for f in ['whole_module.pt']:
    o = {}; flat(dict(torch.load(P(f), weights_only=False).state_dict()), '', o); full[f] = o
o = {}; flat(dict(torch.jit.load(P('scripted.pt')).state_dict()), '', o); full['scripted.pt'] = o
json.dump(full, open(P('expected.json'), 'w'), indent=1)
print('ok', sorted(os.listdir(out)))
