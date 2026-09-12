"""
Debug TCMN train-vs-inference discrepancy.

Tests:
1. Load TCMN checkpoint, check BatchNorm/InstanceNorm running stats
2. Run forward on a real validation patch (same as training) → check dice
3. Run forward on same patch in .eval() mode → check dice
4. Check if predictor's load_state_dict is corrupting the SegOnlyWrapper
5. Compare softmax output distributions between train and eval mode
"""
import sys, os, json, torch
import numpy as np

BASE = '/scratch/siyavash/Alireza_thesis/external_dataset/rsna-intracranial-aneurysm-detection/brain_external_dataset'
sys.path.insert(0, os.path.join(BASE, 'training'))

os.environ.setdefault('nnUNet_raw', os.path.join(BASE, 'preprocessed/nnUNet_raw'))
os.environ.setdefault('nnUNet_preprocessed', os.path.join(BASE, 'preprocessed/nnUNet_preprocessed'))
os.environ.setdefault('nnUNet_results', os.path.join(BASE, 'results/nnUNet'))

from nnUNetTrainerTCMN import nnUNetTrainerTCMN
import tcmn

ds_name = "Dataset104_TopCoW_joint"
plans = json.load(open(os.path.join(os.environ['nnUNet_preprocessed'], ds_name, 'nnUNetPlans.json')))
ds_json = json.load(open(os.path.join(os.environ['nnUNet_preprocessed'], ds_name, 'dataset.json')))

# Load trainer
trainer = nnUNetTrainerTCMN(plans, '3d_fullres', 0, ds_json,
                            unpack_dataset=False, device=torch.device('cuda'))
ckpt_path = os.path.join(
    os.environ['nnUNet_results'],
    f'{ds_name}/nnUNetTrainerTCMN__nnUNetPlans__3d_fullres/fold_0/checkpoint_final.pth'
)
print(f"Loading checkpoint: {ckpt_path}")
trainer.load_checkpoint(ckpt_path)

net = trainer.network

# ================================================================
# TEST 1: Check InstanceNorm running stats vs affine-only
# ================================================================
print("\n" + "="*60)
print("TEST 1: Check norm layer states")
print("="*60)

for name, mod in net.named_modules():
    if isinstance(mod, torch.nn.InstanceNorm3d):
        has_running = mod.running_mean is not None
        print(f"  {name}: affine={mod.affine}, track_running_stats={mod.track_running_stats}, "
              f"has_running_mean={has_running}")
        if has_running:
            print(f"    running_mean range: [{mod.running_mean.min():.4f}, {mod.running_mean.max():.4f}]")
        break  # just check first one
    if isinstance(mod, tcmn.TCMNLayer):
        fb = mod.fallback_norm
        has_running = fb.running_mean is not None
        print(f"  {name} (TCMNLayer fallback): affine={fb.affine}, "
              f"track_running_stats={fb.track_running_stats}, has_running_mean={has_running}")
        # Check FiLM params
        with torch.no_grad():
            mod_emb = mod.modality_embed.weight  # (2, embed_dim)
            dep_emb = mod.depth_embed.weight     # (5, embed_dim)
            print(f"    modality_embed norm: {mod_emb.norm():.4f}")
            print(f"    depth_embed norm: {dep_emb.norm():.4f}")
            # Test FiLM output for CTA (mod=0) at this depth
            emb_in = torch.cat([mod_emb[0:1], dep_emb[mod.depth_id:mod.depth_id+1]], dim=-1)
            film_out = mod.film_mlp(emb_in)
            gamma = film_out[0, :mod.num_features]
            beta = film_out[0, mod.num_features:]
            print(f"    FiLM gamma (CTA, depth={mod.depth_id}): mean={gamma.mean():.4f}, std={gamma.std():.4f}, "
                  f"range=[{gamma.min():.4f}, {gamma.max():.4f}]")
            print(f"    FiLM beta  (CTA, depth={mod.depth_id}): mean={beta.mean():.4f}, std={beta.std():.4f}")
            # Same for MRA
            emb_in_mr = torch.cat([mod_emb[1:2], dep_emb[mod.depth_id:mod.depth_id+1]], dim=-1)
            film_out_mr = mod.film_mlp(emb_in_mr)
            gamma_mr = film_out_mr[0, :mod.num_features]
            beta_mr = film_out_mr[0, mod.num_features:]
            print(f"    FiLM gamma (MRA, depth={mod.depth_id}): mean={gamma_mr.mean():.4f}, std={gamma_mr.std():.4f}")
            print(f"    FiLM beta  (MRA, depth={mod.depth_id}): mean={beta_mr.mean():.4f}, std={beta_mr.std():.4f}")
        break

# ================================================================
# TEST 2: Load a real preprocessed validation case
# ================================================================
print("\n" + "="*60)
print("TEST 2: Load a real validation case and test forward")
print("="*60)

splits = json.load(open(os.path.join(os.environ['nnUNet_preprocessed'], ds_name, 'splits_final.json')))
val_keys = splits[0]['val']
# Pick a CT and MR case
ct_key = [k for k in val_keys if '_ct_' in k][0]
mr_key = [k for k in val_keys if '_mr_' in k][0]
print(f"CT case: {ct_key}")
print(f"MR case: {mr_key}")

preprocessed_dir = os.path.join(os.environ['nnUNet_preprocessed'], ds_name, 'nnUNetPlans_3d_fullres')

# Load preprocessed data
ct_data = np.load(os.path.join(preprocessed_dir, f"{ct_key}.npz"))['data']
ct_seg = np.load(os.path.join(preprocessed_dir, f"{ct_key}.npz"))['seg'] if 'seg' in np.load(os.path.join(preprocessed_dir, f"{ct_key}.npz")).files else None

print(f"CT data shape: {ct_data.shape}")  # (C, D, H, W)

# Extract a center patch (same size as training)
from nnunetv2.utilities.plans_handling.plans_handler import PlansManager
pm = PlansManager(plans)
cm = pm.get_configuration('3d_fullres')
patch_size = cm.patch_size
print(f"Training patch size: {patch_size}")

# Center crop
d, h, w = ct_data.shape[1:]
pd, ph, pw = patch_size
sd = max(0, (d - pd) // 2)
sh = max(0, (h - ph) // 2)
sw = max(0, (w - pw) // 2)
patch = ct_data[:, sd:sd+pd, sh:sh+ph, sw:sw+pw]
patch_tensor = torch.from_numpy(patch).float().unsqueeze(0).cuda()  # (1, C, D, H, W)
print(f"Patch shape: {patch_tensor.shape}")

if ct_seg is not None:
    seg_patch = ct_seg[:, sd:sd+pd, sh:sh+ph, sw:sw+pw]
    seg_tensor = torch.from_numpy(seg_patch).long().squeeze(0).cuda()
    n_fg = (seg_tensor > 0).sum().item()
    n_total = seg_tensor.numel()
    print(f"GT patch: {n_fg}/{n_total} foreground voxels ({100*n_fg/n_total:.2f}%)")

# ================================================================
# TEST 3: Forward in TRAINING mode (same as validation_step)
# ================================================================
print("\n" + "="*60)
print("TEST 3: Forward in TRAINING mode (like validation_step)")
print("="*60)

net.train()
trainer.set_deep_supervision_enabled(True)
mod_ids = torch.tensor([0], device='cuda')  # CTA

with torch.no_grad(), tcmn.modality_context(mod_ids):
    out_train = net(patch_tensor)

if isinstance(out_train, dict):
    seg_train = out_train['seg']
    if isinstance(seg_train, (list, tuple)):
        seg_train = seg_train[0]
    print(f"Training mode output (dict): seg shape={seg_train.shape}")
else:
    seg_train = out_train
    if isinstance(seg_train, (list, tuple)):
        seg_train = seg_train[0]
    print(f"Training mode output: shape={seg_train.shape}")

# Softmax and argmax
probs_train = torch.softmax(seg_train, dim=1)
pred_train = seg_train.argmax(dim=1)
print(f"  max logit: {seg_train.max():.4f}, min logit: {seg_train.min():.4f}")
print(f"  max prob: {probs_train.max():.4f}")
print(f"  bg prob mean: {probs_train[0, 0].mean():.4f}")
print(f"  fg predicted: {(pred_train > 0).sum().item()}")
print(f"  unique labels: {pred_train.unique().tolist()[:20]}")

# ================================================================
# TEST 4: Forward in EVAL mode (same as inference)
# ================================================================
print("\n" + "="*60)
print("TEST 4: Forward in EVAL mode (like inference)")
print("="*60)

net.eval()
trainer.set_deep_supervision_enabled(False)

with torch.no_grad(), tcmn.modality_context(mod_ids):
    out_eval = net(patch_tensor)

if isinstance(out_eval, dict):
    seg_eval = out_eval['seg']
else:
    seg_eval = out_eval
if isinstance(seg_eval, (list, tuple)):
    seg_eval = seg_eval[0]

print(f"Eval mode output: shape={seg_eval.shape}")
probs_eval = torch.softmax(seg_eval, dim=1)
pred_eval = seg_eval.argmax(dim=1)
print(f"  max logit: {seg_eval.max():.4f}, min logit: {seg_eval.min():.4f}")
print(f"  max prob: {probs_eval.max():.4f}")
print(f"  bg prob mean: {probs_eval[0, 0].mean():.4f}")
print(f"  fg predicted: {(pred_eval > 0).sum().item()}")
print(f"  unique labels: {pred_eval.unique().tolist()[:20]}")

# ================================================================
# TEST 5: Compare train vs eval mode outputs
# ================================================================
print("\n" + "="*60)
print("TEST 5: Train vs Eval mode comparison")
print("="*60)
diff = (seg_train - seg_eval).abs()
print(f"  Max absolute difference: {diff.max():.6f}")
print(f"  Mean absolute difference: {diff.mean():.6f}")
if diff.max() < 1e-4:
    print("  >>> IDENTICAL outputs in train vs eval mode")
else:
    print("  >>> DIFFERENT outputs in train vs eval mode!")
    # Check which spatial locations differ most
    max_diff_flat = diff.view(-1).argmax()
    print(f"  Location of max diff: flat index {max_diff_flat}")
    # Check logit distributions
    print(f"  Train logits: mean={seg_train.mean():.4f}, std={seg_train.std():.4f}")
    print(f"  Eval logits:  mean={seg_eval.mean():.4f}, std={seg_eval.std():.4f}")

# ================================================================
# TEST 6: Check if the issue is in normalization behavior
#          (InstanceNorm behaves differently train vs eval IF track_running_stats=True)
# ================================================================
print("\n" + "="*60)
print("TEST 6: InstanceNorm track_running_stats check")
print("="*60)
count_tracked = 0
count_untracked = 0
for name, mod in net.named_modules():
    if isinstance(mod, torch.nn.InstanceNorm3d):
        if mod.track_running_stats:
            count_tracked += 1
        else:
            count_untracked += 1
print(f"  InstanceNorm3d layers: {count_tracked} tracked, {count_untracked} untracked")
if count_tracked > 0:
    print("  >>> WARNING: Some InstanceNorm3d layers track running stats!")
    print("  >>> This causes train/eval behavioral difference!")

# ================================================================
# TEST 7: Compute actual dice on this patch (like pseudo-dice)
# ================================================================
if ct_seg is not None:
    print("\n" + "="*60)
    print("TEST 7: Dice on center patch")
    print("="*60)

    pred = pred_eval.cpu().numpy().squeeze()
    gt = seg_tensor.cpu().numpy()

    # Overall foreground dice
    pred_fg = pred > 0
    gt_fg = gt > 0
    intersection = (pred_fg & gt_fg).sum()
    dice_fg = 2 * intersection / (pred_fg.sum() + gt_fg.sum() + 1e-8)
    print(f"  Foreground dice (binary): {dice_fg:.4f}")
    print(f"  Pred fg voxels: {pred_fg.sum()}, GT fg voxels: {gt_fg.sum()}")

    # Per-class dice
    classes = np.unique(gt)
    classes = classes[classes > 0]
    dices = []
    for c in classes[:5]:  # first 5 classes
        pc = pred == c
        gc = gt == c
        inter = (pc & gc).sum()
        d = 2 * inter / (pc.sum() + gc.sum() + 1e-8)
        dices.append(d)
        print(f"    Class {c}: dice={d:.4f} (pred={pc.sum()}, gt={gc.sum()})")
    print(f"  Mean per-class dice (first {len(dices)}): {np.mean(dices):.4f}")

# ================================================================
# TEST 8: Now load BASELINE checkpoint and compare
# ================================================================
print("\n" + "="*60)
print("TEST 8: Compare with baseline model on same patch")
print("="*60)

from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer

baseline_trainer = nnUNetTrainer(plans, '3d_fullres', 0, ds_json,
                                 unpack_dataset=False, device=torch.device('cuda'))
baseline_ckpt = os.path.join(
    os.environ['nnUNet_results'],
    f'{ds_name}/nnUNetTrainer__nnUNetPlans__3d_fullres/fold_0/checkpoint_final.pth'
)
baseline_trainer.initialize()
baseline_trainer.load_checkpoint(baseline_ckpt)
baseline_trainer.set_deep_supervision_enabled(False)
baseline_trainer.network.eval()

with torch.no_grad():
    baseline_out = baseline_trainer.network(patch_tensor)
if isinstance(baseline_out, (list, tuple)):
    baseline_out = baseline_out[0]

pred_baseline = baseline_out.argmax(dim=1)
probs_baseline = torch.softmax(baseline_out, dim=1)
print(f"  Baseline max logit: {baseline_out.max():.4f}, min: {baseline_out.min():.4f}")
print(f"  Baseline bg prob mean: {probs_baseline[0, 0].mean():.4f}")
print(f"  Baseline fg predicted: {(pred_baseline > 0).sum().item()}")

if ct_seg is not None:
    pred_b = pred_baseline.cpu().numpy().squeeze()
    pred_fg_b = pred_b > 0
    inter_b = (pred_fg_b & gt_fg).sum()
    dice_b = 2 * inter_b / (pred_fg_b.sum() + gt_fg.sum() + 1e-8)
    print(f"  Baseline foreground dice: {dice_b:.4f}")
    print(f"  TCMN foreground dice:     {dice_fg:.4f}")

print("\n=== Debug complete ===")
