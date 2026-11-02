import numpy as np, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False})
T = ["Qatar", "NLM", "TBX11K", "Pakistan", "CIDRZ"]
# fig1: LOCO AUROC (loco_final_summary.md; VinDr column = weighted AUROC from results/vindr_eval.md)
M = {
 "ViT ERM":          [.863,.760,.685,.870,.777,.837],
 "ViT sbal":         [.891,.745,.689,.855,.751,.848],
 "ViT +aug+lung":    [.942,.854,.869,.801,.804,.870],
 "ViT +MixStyle":    [.951,.862,.874,.783,.790,.846],
 "ResNet50":         [.932,.802,.862,.669,.772,.864],
 "xrv-DenseNet":     [.913,.771,.820,.732,.807,.825],
 "RAD-DINO":         [.968,.916,.913,.794,.809,.903],
 "RAD-DINO+MixStyle":[.967,.909,.923,.808,.805,.917],
}
fig, ax = plt.subplots(figsize=(7.6, 3.3)); w = .1
cm = plt.cm.tab10
for i, (k, v) in enumerate(M.items()):
    ax.bar(np.arange(6) + (i - 3.5) * w, v, w, label=k, color=cm(i))
ax.set_xticks(range(6)); ax.set_xticklabels(["Qatar", "NLM", "TBX11K", "Pakistan", "CIDRZ\n(external)", "VinDr\n(external, weighted)"])
ax.set_ylim(.5, 1); ax.set_ylabel("held-out AUROC"); ax.legend(ncol=4, fontsize=6.5, loc="upper center", bbox_to_anchor=(.5, -.2), frameon=False)
fig.tight_layout(); fig.savefig("fig_loco.pdf")
# fig2: shortcut controls
C = {"lung silhouette only": [.84,.71,.60,.80,.61], "outer border only": [.78,.53,.61,.76,.46], "frozen xrv probe": [.84,.73,.69,.68,.70]}
fig, ax = plt.subplots(figsize=(6, 2.8)); w = .26
for i, (k, v) in enumerate(C.items()): ax.bar(np.arange(5) + (i - 1) * w, v, w, label=k)
ax.axhline(.5, c="k", lw=.6, ls=":"); ax.set_xticks(range(5)); ax.set_xticklabels(T); ax.set_ylim(.4, .9)
ax.set_ylabel("LOCO AUROC"); ax.legend(fontsize=7, frameon=False, ncol=3, loc="upper center", bbox_to_anchor=(.5, 1.15))
fig.tight_layout(); fig.savefig("fig_shortcut.pdf")
# fig3: fusion
arms = ["audio", "text+audio", "text", "image (RAD-DINO\n+ViT ens.)", "text+audio\n+image (MD-MLP)", "text+audio\n+image (stack)", "text+image\n(relw)"]
v = [.610, .775, .782, .827, .800, .838, .851]
fig, ax = plt.subplots(figsize=(6.6, 2.8))
ax.bar(range(7), v, color=["#999"]*3 + ["#4c78a8"] + ["#f58518"]*2 + ["#54a24b"])
for i, x in enumerate(v): ax.text(i, x + .005, f"{x:.3f}", ha="center", fontsize=7)
ax.set_xticks(range(7)); ax.set_xticklabels(arms, fontsize=6.5); ax.set_ylim(.5, .9); ax.set_ylabel("CIDRZ AUROC (10x5-fold CV)")
fig.tight_layout(); fig.savefig("fig_fusion.pdf")
# fig4: calibration slope/ECE for bb_raddino
fig, ax = plt.subplots(figsize=(5, 2.6))
ece = [.035,.121,.237,.197,.487]; sl = [.56,.44,.27,.20,.27]
ax.bar(np.arange(5) - .2, ece, .4, label="ECE"); ax.bar(np.arange(5) + .2, sl, .4, label="calibration slope (ideal 1)")
ax.set_xticks(range(5)); ax.set_xticklabels(T); ax.legend(frameon=False, fontsize=7); ax.set_title("RAD-DINO baseline", fontsize=8)
fig.tight_layout(); fig.savefig("fig_calib.pdf")
