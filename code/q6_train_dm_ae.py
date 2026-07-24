"""Marco follow-up #6: SE(3)-invariant distance-matrix autoencoder.

Trains molearn's CNN2d_AE (encoder sees only the pairwise Ca-Ca distance matrix,
so the latent is rotation/translation-invariant by construction) on the 6,531
v9.1 activation loops (27 Ca each). Loss is on the DISTANCE MATRIX of the
reconstruction vs input (also SE(3)-invariant), so the whole pipeline is frame
independent -- unlike the coordinate FoldingNet, whose z0 was dominated by loop
placement (Marco Q1).

Model classes copied verbatim from molearn.models.CNN2d_AE (pure torch), so no
molearn install is needed -- runs on any torch>=2 CUDA env.

Outputs: q6_dm_latent.csv (idx, z0, z1) + q6_dm_ae.ckpt.
"""
import argparse, numpy as np, torch
from torch import nn

# ----------------------- molearn CNN2d_AE (verbatim) -----------------------
class Encoder(nn.Module):
    def __init__(self, latent_dim, dims, channels):
        super().__init__()
        self.convs = nn.ModuleList()
        for in_ch, out_ch in zip(channels[:-1], channels[1:]):
            self.convs.append(nn.Sequential(
                nn.Conv2d(in_ch, out_ch, 4, 2, 1, bias=True),
                nn.BatchNorm2d(out_ch), nn.LeakyReLU(0.1, inplace=True)))
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.finallayer = nn.Linear(channels[-1], latent_dim)
    def forward(self, x):
        for conv in self.convs: x = conv(x)
        x = self.global_pool(x).view(x.size(0), -1)
        return self.finallayer(x)

class Decoder(nn.Module):
    def __init__(self, latent_dim, dims, channels):
        super().__init__()
        self.from_latent = nn.Linear(latent_dim, channels[-1]*dims[-1])
        self.dims = dims; self.channels = channels
        dims_rev = dims[::-1]; ch_rev = channels[::-1]
        layers = []
        for i in range(len(dims_rev)-1):
            h_in, h_out = dims_rev[i], dims_rev[i+1]
            in_ch = ch_rev[i]; default_out = ch_rev[i+1]
            is_last = (i == len(dims_rev)-2)
            out_ch = 3 if is_last else default_out
            op_h = h_out - 2*h_in
            layers.append(nn.ConvTranspose1d(in_ch, out_ch, 4, 2, 1, op_h, bias=True))
            if not is_last:
                layers += [nn.BatchNorm1d(out_ch), nn.LeakyReLU(0.1, inplace=True)]
        self.convs = nn.Sequential(*layers)
    def forward(self, z):
        z = z.view(z.size(0), -1)
        h = self.from_latent(z).view(z.size(0), -1, self.dims[-1])
        return self.convs(h)

class AutoEncoder(nn.Module):
    def __init__(self, dm_dim, latent_dim=2, init_c=32, m=2, min_size=9):
        super().__init__()
        dims, channels = self._dc(dm_dim, init_c, m, min_size)
        print(f"dims={dims} channels={channels}")
        self.dims = dims; self.channels = channels
        self.encoder = Encoder(latent_dim, dims, channels)
        self.decoder = Decoder(latent_dim, dims, channels)
    def _dc(self, dm_dim, init_c, m, min_size):
        dims=[dm_dim]; channels=[1]; curr=dm_dim; ch=init_c
        while curr >= min_size:
            channels.append(ch); curr=(curr+2-4)//2+1; dims.append(curr); ch=int(ch*m)
        return dims, channels
    @staticmethod
    def coords_to_dm(coord):
        n = coord.size(1)
        G = torch.bmm(coord, coord.transpose(1, 2))
        Gt = torch.diagonal(G, dim1=-2, dim2=-1)[:, None, :].repeat(1, n, 1)
        dm = torch.clamp(Gt + Gt.transpose(1, 2) - 2*G, min=1e-12)
        return torch.sqrt(dm)[:, None, :, :]
    def encode(self, x): return self.encoder(self.coords_to_dm(x))
    def decode(self, z): return self.decoder(z).squeeze(-1).permute(0, 2, 1)
    def forward(self, x): return self.decode(self.encode(x))

# ----------------------- data + training -----------------------
def load_pdb(path):
    out, cur = [], []
    for line in open(path):
        if line.startswith("MODEL"): cur = []
        elif line.startswith("ATOM") and line[12:16].strip() == "CA":
            cur.append([float(line[30:38]), float(line[38:46]), float(line[46:54])])
        elif line.startswith("ENDMDL") and cur: out.append(cur); cur = []
    return np.asarray(out, np.float32)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--combined-pdb", required=True)
    ap.add_argument("--out-latent", required=True)
    ap.add_argument("--out-ckpt", required=True)
    ap.add_argument("--epochs", type=int, default=300)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--seed", type=int, default=25)
    a = ap.parse_args()
    torch.manual_seed(a.seed); np.random.seed(a.seed)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("device", dev, torch.cuda.get_device_name(0) if torch.cuda.is_available() else "")

    X = load_pdb(a.combined_pdb)
    n, natoms, _ = X.shape
    mu, sd = X.mean(), X.std()
    Xs = torch.tensor((X - mu) / sd, dtype=torch.float32)     # scale (dm is frame-invariant)
    print(f"loops {n} x {natoms} Ca ; global mean {mu:.3f} std {sd:.3f}")

    perm = np.random.default_rng(a.seed).permutation(n)
    ntest = n // 10
    te, tr = perm[:ntest], perm[ntest:]
    Xtr = Xs[tr].to(dev)

    net = AutoEncoder(dm_dim=natoms, latent_dim=2).to(dev)
    opt = torch.optim.Adam(net.parameters(), lr=1e-3)
    mse = nn.MSELoss()
    Xte = Xs[te].to(dev)
    import copy
    best_val = float('inf'); best_state = None
    for ep in range(a.epochs):
        net.train(); idx = torch.randperm(len(Xtr))
        tot = 0.0
        for i in range(0, len(Xtr), a.batch):
            b = Xtr[idx[i:i+a.batch]]
            dm_in = AutoEncoder.coords_to_dm(b)
            dm_out = AutoEncoder.coords_to_dm(net.decode(net.encode(b)))
            loss = mse(dm_out, dm_in)
            opt.zero_grad(); loss.backward(); opt.step()
            tot += loss.item()*len(b)
        if ep % 10 == 0 or ep == a.epochs-1:
            net.eval()
            with torch.no_grad():
                vdm_in = AutoEncoder.coords_to_dm(Xte)
                vdm_out = AutoEncoder.coords_to_dm(net.decode(net.encode(Xte)))
                vloss = mse(vdm_out, vdm_in).item()
            if vloss < best_val:
                best_val = vloss; best_state = copy.deepcopy(net.state_dict())
            if ep % 50 == 0 or ep == a.epochs-1:
                print(f"epoch {ep:4d}  train {tot/len(Xtr):.5f}  val_dm_mse {vloss:.5f}  best {best_val:.5f}", flush=True)

    if best_state is not None: net.load_state_dict(best_state)   # use best-val model
    print(f"BEST_VAL_DM_MSE {best_val:.5f}", flush=True)
    net.eval()
    Z = []
    with torch.no_grad():
        for i in range(0, n, 256):
            Z.append(net.encode(Xs[i:i+256].to(dev)).cpu().numpy())
    Z = np.concatenate(Z, 0)
    import csv
    with open(a.out_latent, "w", newline="") as fh:
        w = csv.writer(fh); w.writerow(["idx", "z0", "z1"])
        for i in range(n): w.writerow([i, Z[i, 0], Z[i, 1]])
    torch.save({"model_state_dict": net.state_dict(), "mu": float(mu), "sd": float(sd)}, a.out_ckpt)
    print(f"wrote {a.out_latent}  z0 std {Z[:,0].std():.3f}  z1 std {Z[:,1].std():.3f}")

if __name__ == "__main__":
    main()
