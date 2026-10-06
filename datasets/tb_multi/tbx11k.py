"""TBX11K loader (Liu et al., CVPR 2020; CC BY 4.0), Kaggle mirror usmanshams/tbx-11.

Labelled images: imgs/health (3800, label 0), imgs/sick (3800, "sick but non-TB", label 0),
imgs/tb (800). TB images are split by their box annotations: any ActiveTuberculosis box -> label 1
(`category=active`); only ObsoletePulmonaryTuberculosis boxes -> `category=latent`, and no boxes ->
`category=uncertain`; both are excluded from the binary task (`use_for_binary=False`). imgs/test
(3302) has no public labels and imgs/extra holds Montgomery/Shenzhen/DA/DB copies, so both are
skipped (extra would duplicate the Montgomery/Shenzhen corpora). `boxes` = JSON list of
[class, x0, y0, x1, y1] as fractions of the original image size.
"""
import glob
import json
import os
import xml.etree.ElementTree as ET

import numpy as np

ROOT = os.path.expanduser("~/.cache/kagglehub/datasets/usmanshams/tbx-11/versions/1/TBX11K")


def _boxes(stem):
    p = os.path.join(ROOT, "annotations", "xml", stem + ".xml")
    if not os.path.exists(p):
        return []
    r = ET.parse(p).getroot()
    w, h = float(r.find("size/width").text), float(r.find("size/height").text)
    out = []
    for o in r.findall("object"):
        b = o.find("bndbox")
        out.append([o.find("name").text] + [float(b.find(k).text) / s for k, s in
                                            (("xmin", w), ("ymin", h), ("xmax", w), ("ymax", h))])
    return out


def build(from_paths):
    paths, labels, cat, use, boxes = [], [], [], [], []
    for folder, c in (("health", "healthy"), ("sick", "sick")):
        for p in sorted(glob.glob(os.path.join(ROOT, "imgs", folder, "*.png"))):
            paths.append(p); labels.append(0); cat.append(c); use.append(True); boxes.append("[]")
    for p in sorted(glob.glob(os.path.join(ROOT, "imgs", "tb", "*.png"))):
        b = _boxes(os.path.splitext(os.path.basename(p))[0])
        names = {x[0] for x in b}
        c = "active" if "ActiveTuberculosis" in names else ("latent" if names else "uncertain")
        paths.append(p); labels.append(int(c == "active")); cat.append(c); use.append(c == "active")
        boxes.append(json.dumps(b))
    return from_paths("tbx11k", paths, labels,
                      extra=dict(category=np.array(cat), use_for_binary=np.array(use), boxes=np.array(boxes)))
