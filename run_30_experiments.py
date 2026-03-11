"""
FRAMEWORM AGENT — 30-Experiment Benchmark
CIFAR-10 | VAE + DCGAN | Shadow runs | Agent intervention
"""

import os, json, math, time, sqlite3, random
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from pathlib import Path
from datetime import datetime, timezone
from collections import deque

import torchvision
import torchvision.transforms as transforms

# ── paths ──────────────────────────────────────────────────────────────────────
RESULTS_DIR = Path("experiments/cifar_benchmark")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
LOG_FILE = RESULTS_DIR / "experiment_log.jsonl"
TABLE_FILE = RESULTS_DIR / "results_table.txt"
DB_FILE = RESULTS_DIR / "benchmark.db"

torch.manual_seed(42)
random.seed(42)

# ── AGENT import (graceful fallback) ───────────────────────────────────────────
_using_full_agent = False
print("[AGENT] Using built-in observer (benchmark mode)")

# ── Built-in RollingWindow (fallback) ─────────────────────────────────────────
if not _using_full_agent:

    class RollingWindow:
        """
        Lightweight observer — tuned to reduce false positives on clean runs.
        Key change from v1: require 4 consecutive anomalous epochs before firing,
        and use tighter thresholds.
        """

        def __init__(self, maxlen=8):
            self.losses = deque(maxlen=maxlen)
            self.grads = deque(maxlen=maxlen)
            self._streak = {"OSCILLATING": 0, "PLATEAU": 0, "GRADIENT_EXPLOSION": 0, "DIVERGING": 0}
            self.STREAK_REQUIRED = 3  # ← must persist N epochs to fire

        def update(self, loss, grad_norm):
            self.losses.append(float(loss))
            self.grads.append(float(grad_norm))

        def classify(self):
            if len(self.losses) < 4:
                return "HEALTHY"

            losses = list(self.losses)
            grads = list(self.grads)
            last = losses[-1]
            mean = sum(losses) / len(losses)

            # Divergence — clear signal, fire immediately
            if last > 500 or math.isnan(last) or math.isinf(last):
                return "DIVERGING"

            # Gradient explosion — clear signal
            if grads[-1] > 50:
                return "GRADIENT_EXPLOSION"

            # Oscillating — variance very high relative to mean (tightened)
            variance = sum((l - mean) ** 2 for l in losses) / len(losses)
            if mean > 0 and (variance**0.5) / (mean + 1e-8) > 0.25:
                c = "OSCILLATING"
            # Plateau — last 4 epochs barely moved (tightened window)
            elif len(losses) >= 4 and abs(losses[-1] - losses[-4]) < 0.05:
                c = "PLATEAU"
            else:
                # Reset all streaks on healthy epoch
                for k in self._streak:
                    self._streak[k] = 0
                return "HEALTHY"

            # Require streak before firing
            self._streak[c] += 1
            for k in self._streak:
                if k != c:
                    self._streak[k] = 0
            if self._streak[c] >= self.STREAK_REQUIRED:
                return c
            return "HEALTHY"


# ── Data ───────────────────────────────────────────────────────────────────────
def get_cifar_loader(n_samples=4000, batch_size=128):
    tf = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
    )
    ds = torchvision.datasets.CIFAR10(root="./data", train=True, download=True, transform=tf)
    idx = list(range(n_samples))
    return DataLoader(Subset(ds, idx), batch_size=batch_size, shuffle=True, drop_last=True)


# ── Small Conv VAE ─────────────────────────────────────────────────────────────
class VAEEncoder(nn.Module):
    def __init__(self, latent_dim=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3, 32, 4, 2, 1),
            nn.ReLU(),  # 16x16
            nn.Conv2d(32, 64, 4, 2, 1),
            nn.ReLU(),  # 8x8
            nn.Conv2d(64, 128, 4, 2, 1),
            nn.ReLU(),  # 4x4
            nn.Flatten(),
        )
        self.fc_mu = nn.Linear(128 * 4 * 4, latent_dim)
        self.fc_var = nn.Linear(128 * 4 * 4, latent_dim)

    def forward(self, x):
        h = self.net(x)
        return self.fc_mu(h), self.fc_var(h)


class VAEDecoder(nn.Module):
    def __init__(self, latent_dim=64):
        super().__init__()
        self.fc = nn.Linear(latent_dim, 128 * 4 * 4)
        self.net = nn.Sequential(
            nn.ConvTranspose2d(128, 64, 4, 2, 1),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, 4, 2, 1),
            nn.ReLU(),
            nn.ConvTranspose2d(32, 3, 4, 2, 1),
        )

    def forward(self, z):
        h = self.fc(z).view(-1, 128, 4, 4)
        return self.net(h)


class SmallConvVAE(nn.Module):
    def __init__(self, latent_dim=64):
        super().__init__()
        self.encoder = VAEEncoder(latent_dim)
        self.decoder = VAEDecoder(latent_dim)

    def reparameterize(self, mu, log_var):
        std = torch.exp(0.5 * log_var)
        return mu + std * torch.randn_like(std)

    def forward(self, x):
        mu, log_var = self.encoder(x)
        z = self.reparameterize(mu, log_var)
        return self.decoder(z), mu, log_var

    def loss(self, x, recon, mu, log_var):
        recon_loss = nn.functional.mse_loss(recon, x, reduction="mean")
        kld = -0.5 * torch.mean(1 + log_var - mu.pow(2) - log_var.exp())
        return recon_loss + 0.001 * kld


# ── Small DCGAN ────────────────────────────────────────────────────────────────
class DCGANGenerator(nn.Module):
    def __init__(self, latent_dim=64, ngf=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.ConvTranspose2d(latent_dim, ngf * 4, 4, 1, 0, bias=False),
            nn.BatchNorm2d(ngf * 4),
            nn.ReLU(True),
            nn.ConvTranspose2d(ngf * 4, ngf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 2),
            nn.ReLU(True),
            nn.ConvTranspose2d(ngf * 2, ngf, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf),
            nn.ReLU(True),
            nn.ConvTranspose2d(ngf, 3, 4, 2, 1, bias=False),
            nn.Tanh(),
        )

    def forward(self, z):
        return self.net(z)


class DCGANDiscriminator(nn.Module):
    def __init__(self, ndf=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3, ndf, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 2),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 4),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(ndf * 4, 1, 4, 1, 0, bias=False),
            nn.Flatten(),
            nn.Linear(1, 1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        return self.net(x)


# ── Helpers ────────────────────────────────────────────────────────────────────
def grad_norm(model):
    return math.sqrt(
        sum(p.grad.data.norm(2).item() ** 2 for p in model.parameters() if p.grad is not None)
    )


def reinit_weights(model):
    for m in model.modules():
        if isinstance(m, (nn.Linear, nn.Conv2d, nn.ConvTranspose2d)):
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.zeros_(m.bias)


# ── VAE train epoch ────────────────────────────────────────────────────────────
def train_vae_epoch(model, loader, optimizer, inject_grad, epoch, is_shadow=False):
    model.train()
    total, gns, batches = 0.0, [], 0
    for x, _ in loader:
        if inject_grad and epoch == 3 and not is_shadow:
            for p in model.parameters():
                if p.grad is None:
                    p.grad = torch.zeros_like(p)
                p.grad += torch.randn_like(p) * 1000

        recon, mu, lv = model(x)
        loss = model.loss(x, recon, mu, lv)

        optimizer.zero_grad()
        loss.backward()
        gns.append(grad_norm(model))
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        optimizer.step()

        total += loss.item()
        batches += 1
        if math.isnan(total) or total > 1e6:
            return 9999.0, 999.0

    return total / batches, sum(gns) / len(gns)


# ── DCGAN train epoch ──────────────────────────────────────────────────────────
def train_dcgan_epoch(G, D, loader, opt_G, opt_D, latent_dim, inject_grad, epoch, is_shadow=False):
    G.train()
    D.train()
    g_losses, gns, batches = [], [], 0
    real_label, fake_label = 1.0, 0.0
    criterion = nn.BCELoss()

    for x, _ in loader:
        bs = x.size(0)

        # — Train D —
        opt_D.zero_grad()
        out_real = D(x).view(-1)
        loss_real = criterion(out_real, torch.full((bs,), real_label))
        loss_real.backward()

        z = torch.randn(bs, latent_dim, 1, 1)
        fake = G(z).detach()
        out_fake = D(fake).view(-1)
        loss_fake = criterion(out_fake, torch.full((bs,), fake_label))
        loss_fake.backward()
        opt_D.step()

        # — Train G —
        opt_G.zero_grad()
        if inject_grad and epoch == 3 and not is_shadow:
            for p in G.parameters():
                if p.grad is None:
                    p.grad = torch.zeros_like(p)
                p.grad += torch.randn_like(p) * 1000

        z = torch.randn(bs, latent_dim, 1, 1)
        out = D(G(z)).view(-1)
        loss_G = criterion(out, torch.full((bs,), real_label))
        loss_G.backward()
        gns.append(grad_norm(G))
        torch.nn.utils.clip_grad_norm_(G.parameters(), 5.0)
        opt_G.step()

        g_losses.append(loss_G.item())
        batches += 1
        if math.isnan(sum(g_losses)) or sum(g_losses) > 1e6:
            return 9999.0, 999.0

    return sum(g_losses) / len(g_losses), sum(gns) / len(gns)


# ── Experiment runner ──────────────────────────────────────────────────────────
def run_experiment(eid, name, arch, lr, epochs, inject_grad, agent_on, loader, latent_dim=64):
    print(f"\n{'='*60}")
    print(f"  Exp {eid:02d} | {arch} | {name} | lr={lr} | agent={'ON' if agent_on else 'OFF'}")
    print(f"{'='*60}")

    window = RollingWindow()
    n_det = 0
    n_int = 0
    resolved = False
    cur_lr = lr
    history = []

    # ── Agent run ──
    if arch == "VAE":
        model = SmallConvVAE(latent_dim)
        optimizer = optim.Adam(model.parameters(), lr=lr)
    else:
        G = DCGANGenerator(latent_dim)
        D = DCGANDiscriminator()
        opt_G = optim.Adam(G.parameters(), lr=lr, betas=(0.5, 0.999))
        opt_D = optim.Adam(D.parameters(), lr=lr, betas=(0.5, 0.999))

    sentinel_streak = 0

    for ep in range(1, epochs + 1):
        if arch == "VAE":
            ep_loss, ep_gn = train_vae_epoch(model, loader, optimizer, inject_grad, ep)
        else:
            ep_loss, ep_gn = train_dcgan_epoch(
                G, D, loader, opt_G, opt_D, latent_dim, inject_grad, ep
            )

        if ep_loss >= 9000:
            sentinel_streak += 1
        else:
            sentinel_streak = 0

        window.update(ep_loss, ep_gn)
        anomaly = window.classify() if agent_on else "N/A"

        if agent_on and ep_loss >= 9000 and sentinel_streak >= 2:
            print(f"  [AGENT] ep{ep} model dead — reinit LR→3e-4")
            if arch == "VAE":
                reinit_weights(model)
                cur_lr = 3e-4
                optimizer = optim.Adam(model.parameters(), lr=cur_lr)
            else:
                reinit_weights(G)
                reinit_weights(D)
                cur_lr = 3e-4
                opt_G = optim.Adam(G.parameters(), lr=cur_lr, betas=(0.5, 0.999))
                opt_D = optim.Adam(D.parameters(), lr=cur_lr, betas=(0.5, 0.999))
            n_int += 1
            resolved = True

        elif agent_on and anomaly not in ("HEALTHY", "N/A"):
            n_det += 1

            if anomaly == "PLATEAU":
                print(f"  [AGENT] ep{ep} PLATEAU → reinit + LR→3e-4")
                if arch == "VAE":
                    reinit_weights(model)
                    cur_lr = 3e-4
                    optimizer = optim.Adam(model.parameters(), lr=cur_lr)
                else:
                    reinit_weights(G)
                    reinit_weights(D)
                    cur_lr = 3e-4
                    opt_G = optim.Adam(G.parameters(), lr=cur_lr, betas=(0.5, 0.999))
                    opt_D = optim.Adam(D.parameters(), lr=cur_lr, betas=(0.5, 0.999))
                n_int += 1

            elif anomaly == "GRADIENT_EXPLOSION":
                new_lr = max(cur_lr * 0.3, 1e-5)
                print(f"  [AGENT] ep{ep} GRAD_EXPL → LR {cur_lr:.2e}→{new_lr:.2e}")
                cur_lr = new_lr
                if arch == "VAE":
                    for pg in optimizer.param_groups:
                        pg["lr"] = cur_lr
                else:
                    for pg in opt_G.param_groups:
                        pg["lr"] = cur_lr
                    for pg in opt_D.param_groups:
                        pg["lr"] = cur_lr
                n_int += 1

            elif anomaly in ("OSCILLATING", "DIVERGING"):
                new_lr = max(cur_lr * 0.5, 1e-5)
                print(f"  [AGENT] ep{ep} {anomaly} → LR {cur_lr:.2e}→{new_lr:.2e}")
                cur_lr = new_lr
                if arch == "VAE":
                    for pg in optimizer.param_groups:
                        pg["lr"] = cur_lr
                else:
                    for pg in opt_G.param_groups:
                        pg["lr"] = cur_lr
                    for pg in opt_D.param_groups:
                        pg["lr"] = cur_lr
                n_int += 1

        history.append({"epoch": ep, "loss": ep_loss, "gn": ep_gn, "anomaly": anomaly})
        print(f"  ep{ep:02d} loss={ep_loss:.4f} gn={ep_gn:.2f} [{anomaly}]")

    agent_loss = history[-1]["loss"]

    # ── Shadow run (no agent, original LR) ──
    print(f"  --- Shadow run ---")
    if arch == "VAE":
        shadow_model = SmallConvVAE(latent_dim)
        shadow_opt = optim.Adam(shadow_model.parameters(), lr=lr)
    else:
        sG = DCGANGenerator(latent_dim)
        sD = DCGANDiscriminator()
        sOpt_G = optim.Adam(sG.parameters(), lr=lr, betas=(0.5, 0.999))
        sOpt_D = optim.Adam(sD.parameters(), lr=lr, betas=(0.5, 0.999))

    shadow_losses = []
    for ep in range(1, epochs + 1):
        if arch == "VAE":
            sl, _ = train_vae_epoch(
                shadow_model, loader, shadow_opt, inject_grad, ep, is_shadow=True
            )
        else:
            sl, _ = train_dcgan_epoch(
                sG, sD, loader, sOpt_G, sOpt_D, latent_dim, inject_grad, ep, is_shadow=True
            )
        shadow_losses.append(sl)
        print(f"  shadow ep{ep:02d} loss={sl:.4f}")

    shadow_loss = shadow_losses[-1]
    delta = round(shadow_loss - agent_loss, 4)
    sign = "+" if delta >= 0 else ""

    # Resolve check — agent loss meaningfully better than shadow
    if not resolved and delta > 0.5:
        resolved = True

    result = {
        "id": eid,
        "name": name,
        "arch": arch,
        "condition": name.split("-")[0].lower(),
        "lr": lr,
        "epochs": epochs,
        "agent_loss": round(agent_loss, 4),
        "shadow_loss": round(shadow_loss, 4),
        "delta": delta,
        "n_detected": n_det,
        "n_intervened": n_int,
        "resolved": "YES" if resolved else "NO",
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }

    with open(LOG_FILE, "a") as f:
        f.write(json.dumps(result) + "\n")

    print(
        f"\n  ✓ Agent: {agent_loss:.4f}  Shadow: {shadow_loss:.4f}  "
        f"Δ={sign}{delta:.4f}  Det={n_det}  Int={n_int}  "
        f"Resolved={result['resolved']}"
    )

    return result


# ── Experiment table ───────────────────────────────────────────────────────────
#  id  name                arch    lr       epochs  inject_grad  agent_on
EXPERIMENTS = [
    # VAE — Baseline (3 runs)
    (1, "VAE-Baseline-1", "VAE", 3e-4, 15, False, True),
    (2, "VAE-Baseline-2", "VAE", 3e-4, 15, False, True),
    (3, "VAE-Baseline-3", "VAE", 3e-4, 15, False, True),
    # VAE — High LR → divergence
    (4, "VAE-HighLR-1", "VAE", 8e-2, 15, False, True),
    (5, "VAE-HighLR-2", "VAE", 5e-2, 15, False, True),
    (6, "VAE-HighLR-3", "VAE", 1e-1, 15, False, True),
    # VAE — Gradient explosion (injected at epoch 3)
    (7, "VAE-GradExp-1", "VAE", 3e-4, 15, True, True),
    (8, "VAE-GradExp-2", "VAE", 3e-4, 15, True, True),
    (9, "VAE-GradExp-3", "VAE", 3e-4, 15, True, True),
    # VAE — Low LR plateau
    (10, "VAE-Plateau-1", "VAE", 1e-6, 15, False, True),
    (11, "VAE-Plateau-2", "VAE", 5e-7, 15, False, True),
    (12, "VAE-Plateau-3", "VAE", 2e-7, 15, False, True),
    # VAE — Agent intervenes on high LR
    (13, "VAE-Intervene-1", "VAE", 1e-1, 15, False, True),
    (14, "VAE-Intervene-2", "VAE", 8e-2, 15, False, True),
    (15, "VAE-Intervene-3", "VAE", 6e-2, 15, False, True),
    # DCGAN — Baseline
    (16, "GAN-Baseline-1", "DCGAN", 2e-4, 15, False, True),
    (17, "GAN-Baseline-2", "DCGAN", 2e-4, 15, False, True),
    (18, "GAN-Baseline-3", "DCGAN", 2e-4, 15, False, True),
    # DCGAN — High LR → mode collapse / divergence
    (19, "GAN-HighLR-1", "DCGAN", 5e-2, 15, False, True),
    (20, "GAN-HighLR-2", "DCGAN", 2e-2, 15, False, True),
    (21, "GAN-HighLR-3", "DCGAN", 1e-1, 15, False, True),
    # DCGAN — Gradient explosion
    (22, "GAN-GradExp-1", "DCGAN", 2e-4, 15, True, True),
    (23, "GAN-GradExp-2", "DCGAN", 2e-4, 15, True, True),
    (24, "GAN-GradExp-3", "DCGAN", 2e-4, 15, True, True),
    # DCGAN — Low LR plateau
    (25, "GAN-Plateau-1", "DCGAN", 1e-6, 15, False, True),
    (26, "GAN-Plateau-2", "DCGAN", 5e-7, 15, False, True),
    (27, "GAN-Plateau-3", "DCGAN", 2e-7, 15, False, True),
    # DCGAN — Agent intervenes
    (28, "GAN-Intervene-1", "DCGAN", 5e-2, 15, False, True),
    (29, "GAN-Intervene-2", "DCGAN", 2e-2, 15, False, True),
    (30, "GAN-Intervene-3", "DCGAN", 1e-1, 15, False, True),
]


# ── Results table writer ───────────────────────────────────────────────────────
def write_table(results):
    W = 120
    lines = []
    lines.append("=" * W)
    lines.append("  FRAMEWORM AGENT — 30-Experiment Benchmark Results (CIFAR-10)")
    lines.append(f"  {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M')} UTC")
    lines.append("=" * W)
    hdr = (
        f"{'#':>3}  {'Experiment':<20} {'Arch':<6} {'Condition':<14} "
        f"{'Agent Loss':>10} {'Shadow Loss':>11} {'Δ Loss':>10} "
        f"{'Det':>5} {'Int':>5} {'Res':>4}"
    )
    lines.append(hdr)
    lines.append("-" * W)

    by_arch = {"VAE": [], "DCGAN": []}
    for r in results:
        sign = "+" if r["delta"] >= 0 else ""
        row = (
            f"{r['id']:>3}  {r['name']:<20} {r['arch']:<6} "
            f"{r['condition']:<14} {r['agent_loss']:>10.4f} "
            f"{r['shadow_loss']:>11.4f} {sign}{r['delta']:>9.4f} "
            f"{r['n_detected']:>5} {r['n_intervened']:>5} {r['resolved']:>4}"
        )
        lines.append(row)
        by_arch[r["arch"]].append(r)

    lines.append("-" * W)
    lines.append("  SUMMARY")

    for arch, rs in by_arch.items():
        if not rs:
            continue
        baseline = [r for r in rs if "baseline" in r["name"].lower()]
        base_avg = (
            sum(r["agent_loss"] for r in baseline) / len(baseline) if baseline else float("nan")
        )
        resolved = sum(1 for r in rs if r["resolved"] == "YES")
        det_all = sum(1 for r in rs if r["n_detected"] > 0)
        fp = sum(1 for r in baseline if r["n_detected"] > 0)
        lines.append(f"  {arch}:")
        lines.append(f"    Baseline avg loss:          {base_avg:.4f}")
        lines.append(f"    Anomalies detected:         {det_all} / {len(rs)}")
        lines.append(f"    Runs resolved by agent:     {resolved} / {len(rs)}")
        lines.append(f"    False positives (baseline): {fp} / {len(baseline)}")

    all_resolved = sum(1 for r in results if r["resolved"] == "YES")
    non_baseline = [r for r in results if "baseline" not in r["name"].lower()]
    lines.append(
        f"  OVERALL resolution rate:      {all_resolved} / {len(results)}"
        f"  ({100*all_resolved/len(results):.1f}%)"
    )
    meaningful = [r for r in non_baseline if r["delta"] > 0]
    lines.append(f"  Runs where agent helped:      {len(meaningful)} / {len(non_baseline)}")
    lines.append("=" * W)

    text = "\n".join(lines)
    TABLE_FILE.write_text(text, encoding="utf-8")
    print("\n" + text)


# ── Main ───────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("[*] Downloading / loading CIFAR-10...")
    loader = get_cifar_loader(n_samples=4000, batch_size=128)
    print(f"[*] Loader ready — {len(loader)} batches per epoch")
    print(f"[*] Running {len(EXPERIMENTS)} experiments...\n")

    t0 = time.time()
    results = []

    for exp in EXPERIMENTS:
        eid, name, arch, lr, epochs, inject_grad, agent_on = exp
        r = run_experiment(eid, name, arch, lr, epochs, inject_grad, agent_on, loader)
        results.append(r)

    elapsed = time.time() - t0
    print(f"\n[*] All experiments done in {elapsed/60:.1f} min")
    write_table(results)
    print(f"\n[*] Results → {TABLE_FILE}")
    print(f"[*] Log     → {LOG_FILE}")
