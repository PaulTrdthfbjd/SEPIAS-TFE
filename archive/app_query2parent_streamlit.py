# app_query2parent_streamlit.py
import os
import pickle
from pathlib import Path
from collections import defaultdict

import numpy as np
import streamlit as st
from PIL import Image, ImageDraw

import torch
import torch.nn as nn
import torch.nn.functional as F

from scipy.ndimage import label as cc_label, find_objects

from transformers import AutoImageProcessor, AutoModel, CLIPProcessor, CLIPModel

from torchvision.models.detection import (
    maskrcnn_resnet50_fpn_v2,
    MaskRCNN_ResNet50_FPN_V2_Weights,
)

st.set_page_config(page_title="Query → Parent (baseline vs segmentation)", layout="wide")


# -----------------------------
# Canonical paths + normalize
# -----------------------------
def canon(p: str) -> str:
    return os.path.normcase(os.path.normpath(str(p)))

def normalize(v: np.ndarray) -> np.ndarray:
    return v / (np.linalg.norm(v) + 1e-12)

def to_rgb_flat(img: Image.Image, bg=(127, 127, 127)) -> Image.Image:
    # Flatten alpha if present
    if img.mode in ("RGBA", "LA") or (img.mode == "P" and "transparency" in img.info):
        img = img.convert("RGBA")
        bg_img = Image.new("RGBA", img.size, (*bg, 255))
        img = Image.alpha_composite(bg_img, img).convert("RGB")
        return img
    return img.convert("RGB")


# -----------------------------
# Dump loaders
# -----------------------------
@st.cache_data
def load_vec_dump(dump_path: str):
    """
    Accepts:
      - dict[path->vec]
      - payload {'embeddings': dict[path->vec], 'meta': ...}
    Returns: (paths, embs, meta_or_none)
    """
    with open(dump_path, "rb") as f:
        obj = pickle.load(f)

    meta = None
    if isinstance(obj, dict) and "embeddings" in obj:
        emb_dict = obj["embeddings"]
        meta = obj.get("meta", None)
    else:
        emb_dict = obj

    # canon keys
    emb_dict2 = {canon(k): v for k, v in emb_dict.items()}
    paths = list(emb_dict2.keys())
    embs = np.stack([emb_dict2[p] for p in paths], axis=0).astype(np.float32)
    embs = embs / (np.linalg.norm(embs, axis=1, keepdims=True) + 1e-12)

    # canon meta keys + inner paths if any
    meta2 = None
    if isinstance(meta, dict):
        meta2 = {}
        for k, v in meta.items():
            kk = canon(k)
            vv = dict(v)
            if "parent_path" in vv:
                vv["parent_path"] = canon(vv["parent_path"])
            if "crop_path" in vv:
                vv["crop_path"] = canon(vv["crop_path"])
            meta2[kk] = vv

    return paths, embs, meta2


# -----------------------------
# Models (DINO / CLIP)
# -----------------------------
@st.cache_resource
def load_dino():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    proc = AutoImageProcessor.from_pretrained("facebook/dinov2-base")
    model = AutoModel.from_pretrained("facebook/dinov2-base").to(device)
    model.eval()
    return device, proc, model

@st.cache_resource
def load_clip():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    proc = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
    model.eval()
    return device, proc, model

def dino_embed(img: Image.Image, device, proc, model) -> np.ndarray:
    img = to_rgb_flat(img)
    inputs = proc(images=img, return_tensors="pt")
    pixel_values = inputs["pixel_values"].to(device)
    with torch.no_grad():
        out = model(pixel_values)
        emb = out.last_hidden_state.mean(dim=1).squeeze(0).detach().cpu().numpy().astype(np.float32)
    return normalize(emb)

def clip_embed(img: Image.Image, device, proc, model) -> np.ndarray:
    img = to_rgb_flat(img)
    inputs = proc(images=img, return_tensors="pt")
    pixel_values = inputs["pixel_values"].to(device)
    with torch.no_grad():
        emb = model.get_image_features(pixel_values=pixel_values).squeeze(0).detach().cpu().numpy().astype(np.float32)
    return normalize(emb)


# -----------------------------
# SegNet / UNet definitions (same as your scripts)
# -----------------------------
def conv_block(c_in, c_out, n=2):
    layers = []
    for i in range(n):
        layers += [
            nn.Conv2d(c_in if i == 0 else c_out, c_out, 3, padding=1),
            nn.BatchNorm2d(c_out),
            nn.ReLU(inplace=True),
        ]
    return nn.Sequential(*layers)

class SegNetBinary(nn.Module):
    def __init__(self, in_ch=3, out_ch=1):
        super().__init__()
        self.enc1 = conv_block(in_ch, 64, n=2)
        self.pool1 = nn.MaxPool2d(2, 2, return_indices=True)
        self.enc2 = conv_block(64, 128, n=2)
        self.pool2 = nn.MaxPool2d(2, 2, return_indices=True)
        self.enc3 = conv_block(128, 256, n=3)
        self.pool3 = nn.MaxPool2d(2, 2, return_indices=True)
        self.enc4 = conv_block(256, 512, n=3)
        self.pool4 = nn.MaxPool2d(2, 2, return_indices=True)
        self.enc5 = conv_block(512, 512, n=3)
        self.pool5 = nn.MaxPool2d(2, 2, return_indices=True)

        self.unpool5 = nn.MaxUnpool2d(2, 2)
        self.dec5 = conv_block(512, 512, n=3)
        self.unpool4 = nn.MaxUnpool2d(2, 2)
        self.dec4 = conv_block(512, 256, n=3)
        self.unpool3 = nn.MaxUnpool2d(2, 2)
        self.dec3 = conv_block(256, 128, n=3)
        self.unpool2 = nn.MaxUnpool2d(2, 2)
        self.dec2 = conv_block(128, 64, n=2)
        self.unpool1 = nn.MaxUnpool2d(2, 2)
        self.dec1 = conv_block(64, 64, n=2)
        self.out = nn.Conv2d(64, out_ch, 1)

    def forward(self, x):
        x1 = self.enc1(x); s1 = x1.size()
        x, i1 = self.pool1(x1)
        x2 = self.enc2(x); s2 = x2.size()
        x, i2 = self.pool2(x2)
        x3 = self.enc3(x); s3 = x3.size()
        x, i3 = self.pool3(x3)
        x4 = self.enc4(x); s4 = x4.size()
        x, i4 = self.pool4(x4)
        x5 = self.enc5(x); s5 = x5.size()
        x, i5 = self.pool5(x5)

        x = self.unpool5(x, i5, output_size=s5); x = self.dec5(x)
        x = self.unpool4(x, i4, output_size=s4); x = self.dec4(x)
        x = self.unpool3(x, i3, output_size=s3); x = self.dec3(x)
        x = self.unpool2(x, i2, output_size=s2); x = self.dec2(x)
        x = self.unpool1(x, i1, output_size=s1); x = self.dec1(x)
        return self.out(x)

class DoubleConv(nn.Module):
    def __init__(self, c_in, c_out):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(c_in, c_out, 3, padding=1),
            nn.BatchNorm2d(c_out),
            nn.ReLU(inplace=True),
            nn.Conv2d(c_out, c_out, 3, padding=1),
            nn.BatchNorm2d(c_out),
            nn.ReLU(inplace=True),
        )
    def forward(self, x): return self.net(x)

class UNet(nn.Module):
    def __init__(self, in_ch=3, out_ch=1, base=64):
        super().__init__()
        self.down1 = DoubleConv(in_ch, base)
        self.pool1 = nn.MaxPool2d(2)
        self.down2 = DoubleConv(base, base * 2)
        self.pool2 = nn.MaxPool2d(2)
        self.down3 = DoubleConv(base * 2, base * 4)
        self.pool3 = nn.MaxPool2d(2)
        self.down4 = DoubleConv(base * 4, base * 8)
        self.pool4 = nn.MaxPool2d(2)
        self.mid = DoubleConv(base * 8, base * 16)

        self.up4 = nn.ConvTranspose2d(base * 16, base * 8, 2, stride=2)
        self.conv4 = DoubleConv(base * 16, base * 8)
        self.up3 = nn.ConvTranspose2d(base * 8, base * 4, 2, stride=2)
        self.conv3 = DoubleConv(base * 8, base * 4)
        self.up2 = nn.ConvTranspose2d(base * 4, base * 2, 2, stride=2)
        self.conv2 = DoubleConv(base * 4, base * 2)
        self.up1 = nn.ConvTranspose2d(base * 2, base, 2, stride=2)
        self.conv1 = DoubleConv(base * 2, base)
        self.out = nn.Conv2d(base, out_ch, 1)

    def forward(self, x):
        x1 = self.down1(x)
        x2 = self.down2(self.pool1(x1))
        x3 = self.down3(self.pool2(x2))
        x4 = self.down4(self.pool3(x3))
        xm = self.mid(self.pool4(x4))
        y4 = self.up4(xm); y4 = self.conv4(torch.cat([y4, x4], dim=1))
        y3 = self.up3(y4); y3 = self.conv3(torch.cat([y3, x3], dim=1))
        y2 = self.up2(y3); y2 = self.conv2(torch.cat([y2, x2], dim=1))
        y1 = self.up1(y2); y1 = self.conv1(torch.cat([y1, x1], dim=1))
        return self.out(y1)


@st.cache_resource
def load_segnet_ckpt(ckpt_path: str):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = SegNetBinary().to(device)
    ckpt = torch.load(ckpt_path, map_location=device)
    state = ckpt["state_dict"] if isinstance(ckpt, dict) and "state_dict" in ckpt else ckpt
    size = int(ckpt.get("size", 256)) if isinstance(ckpt, dict) else 256
    model.load_state_dict(state, strict=True)
    model.eval()
    return device, model, size

@st.cache_resource
def load_unet_ckpt(ckpt_path: str):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = UNet().to(device)
    ckpt = torch.load(ckpt_path, map_location=device)
    state = ckpt["state_dict"] if isinstance(ckpt, dict) and "state_dict" in ckpt else ckpt
    size = int(ckpt.get("size", 384)) if isinstance(ckpt, dict) else 384
    model.load_state_dict(state, strict=True)
    model.eval()
    return device, model, size


# -----------------------------
# Segmentation helpers (query-time)
# -----------------------------
def apply_mask_rgba(crop_rgb: np.ndarray, mask: np.ndarray) -> Image.Image:
    alpha = (mask.astype(np.uint8) * 255)
    rgba = np.dstack([crop_rgb, alpha])
    return Image.fromarray(rgba, mode="RGBA")

def cc_bboxes_from_mask(mask: np.ndarray, min_area=300, max_objs=10):
    labeled, n_cc = cc_label(mask)
    if n_cc == 0:
        return []
    slices = find_objects(labeled)
    comps = []
    for cc_id, sl in enumerate(slices, start=1):
        if sl is None:
            continue
        cc_mask = (labeled[sl] == cc_id)
        a = int(cc_mask.sum())
        comps.append((a, cc_id, sl))
    comps.sort(reverse=True, key=lambda t: t[0])
    comps = comps[:max_objs]
    out = []
    for a, cc_id, sl in comps:
        if a < min_area:
            continue
        ysl, xsl = sl
        y1, y2 = ysl.start, ysl.stop - 1
        x1, x2 = xsl.start, xsl.stop - 1
        out.append((cc_id, x1, y1, x2, y2))
    return out

def segment_query_maskrcnn(img: Image.Image, score_thr=0.7, max_objs=5, pad=5):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    weights = MaskRCNN_ResNet50_FPN_V2_Weights.DEFAULT
    model = maskrcnn_resnet50_fpn_v2(weights=weights).to(device)
    model.eval()
    preprocess = weights.transforms()

    x = preprocess(to_rgb_flat(img)).to(device)
    with torch.no_grad():
        pred = model([x])[0]

    scores = pred["scores"].detach().cpu().numpy()
    masks = pred["masks"].detach().cpu().numpy()  # (N,1,H,W)

    keep = np.where(scores >= score_thr)[0][:max_objs]
    if len(keep) == 0:
        return []

    img_np = np.array(to_rgb_flat(img))
    H, W = img_np.shape[:2]
    out = []
    for j in keep:
        m = masks[j, 0] > 0.5
        if m.sum() < 100:
            continue
        ys, xs = np.where(m)
        y1, y2 = int(ys.min()), int(ys.max())
        x1, x2 = int(xs.min()), int(xs.max())
        x1 = max(0, x1 - pad); y1 = max(0, y1 - pad)
        x2 = min(W - 1, x2 + pad); y2 = min(H - 1, y2 + pad)

        crop = img_np[y1:y2+1, x1:x2+1]
        crop_mask = m[y1:y2+1, x1:x2+1]
        crop_rgba = apply_mask_rgba(crop, crop_mask)
        out.append({"crop": crop_rgba, "bbox": (x1,y1,x2,y2), "score": float(scores[j])})
    return out

def segment_query_sam_auto(img: Image.Image, sam_ckpt: str, model_type="vit_b",
                           points_per_side=16, pred_iou_thresh=0.86, stability_score_thresh=0.92,
                           crop_n_layers=0, max_masks=15, min_area=400, max_area_ratio=0.65, pad=5):
    try:
        from segment_anything import sam_model_registry, SamAutomaticMaskGenerator
    except Exception as e:
        raise RuntimeError(f"segment-anything not available: {e}")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    sam = sam_model_registry[model_type](checkpoint=sam_ckpt)
    sam.to(device=device)

    mask_generator = SamAutomaticMaskGenerator(
        sam,
        points_per_side=points_per_side,
        pred_iou_thresh=pred_iou_thresh,
        stability_score_thresh=stability_score_thresh,
        crop_n_layers=crop_n_layers,
        output_mode="binary_mask",
    )

    img_rgb = to_rgb_flat(img)
    img_np = np.array(img_rgb)
    H, W = img_np.shape[:2]

    anns = mask_generator.generate(img_np)
    if not anns:
        return []

    anns.sort(key=lambda a: float(a.get("predicted_iou", 0.0)), reverse=True)

    out = []
    kept = 0
    for ann in anns:
        if kept >= max_masks:
            break
        m = ann["segmentation"].astype(bool)
        area = int(m.sum())
        if area < min_area:
            continue
        if (area / float(H*W)) > max_area_ratio:
            continue

        x, y, w, h = ann["bbox"]
        x1, y1 = int(x), int(y)
        x2, y2 = int(x + w - 1), int(y + h - 1)
        x1 = max(0, x1 - pad); y1 = max(0, y1 - pad)
        x2 = min(W - 1, x2 + pad); y2 = min(H - 1, y2 + pad)

        crop = img_np[y1:y2+1, x1:x2+1]
        crop_mask = m[y1:y2+1, x1:x2+1]
        crop_rgba = apply_mask_rgba(crop, crop_mask)
        out.append({"crop": crop_rgba, "bbox": (x1,y1,x2,y2), "score": float(ann.get("predicted_iou", 0.0))})
        kept += 1

    return out

def segment_query_unet_or_segnet(img: Image.Image, method: str, ckpt_path: str,
                                threshold=0.5, input_size=256, min_area=300, max_objs=10, max_area_ratio=0.75):
    img_rgb = to_rgb_flat(img)
    img_np = np.array(img_rgb)
    H, W = img_np.shape[:2]

    if method == "segnet":
        dev, model, default_size = load_segnet_ckpt(ckpt_path)
        input_size = default_size if input_size is None else input_size
    elif method == "unet":
        dev, model, default_size = load_unet_ckpt(ckpt_path)
        input_size = default_size if input_size is None else input_size
    else:
        raise ValueError(method)

    x = torch.from_numpy(img_np).permute(2,0,1).float() / 255.0
    x = x.unsqueeze(0).to(dev)
    x_rs = F.interpolate(x, size=(input_size, input_size), mode="bilinear", align_corners=False)

    with torch.no_grad():
        logits = model(x_rs)
        prob = torch.sigmoid(logits)

    prob = F.interpolate(prob, size=(H, W), mode="bilinear", align_corners=False).squeeze(0).squeeze(0)
    mask = (prob.detach().cpu().numpy() >= threshold)

    area = int(mask.sum())
    if area == 0:
        return []
    if (area / float(H*W)) > max_area_ratio:
        return []

    # connected components
    bbs = cc_bboxes_from_mask(mask, min_area=min_area, max_objs=max_objs)
    out = []
    for cc_id, x1,y1,x2,y2 in bbs:
        crop = img_np[y1:y2+1, x1:x2+1]
        crop_mask = mask[y1:y2+1, x1:x2+1]
        crop_rgba = apply_mask_rgba(crop, crop_mask)
        out.append({"crop": crop_rgba, "bbox": (x1,y1,x2,y2), "score": None})
    return out


# -----------------------------
# Crop→Parent aggregation
# -----------------------------
def draw_bbox(parent_img: Image.Image, bbox_xyxy, width=6):
    img = parent_img.copy().convert("RGB")
    draw = ImageDraw.Draw(img)
    x1, y1, x2, y2 = bbox_xyxy
    for w in range(width):
        draw.rectangle([x1-w, y1-w, x2+w, y2+w], outline=(255, 0, 0))
    return img

def rank_parents_from_crop_scores(crop_paths, crop_scores, meta, agg="max", top_per_parent=3):
    bucket = defaultdict(list)  # parent_path -> list[(score, crop_path, bbox)]
    for p, s in zip(crop_paths, crop_scores):
        rec = meta.get(canon(p)) if meta else None
        if not rec:
            continue
        parent = rec.get("parent_path", None)
        bbox = rec.get("bbox_xyxy", None)
        if not parent:
            continue
        bucket[parent].append((float(s), canon(p), bbox))

    parent_rows = []
    for parent, lst in bucket.items():
        lst.sort(key=lambda t: t[0], reverse=True)
        top_lst = lst[:top_per_parent]
        if agg == "max":
            parent_score = top_lst[0][0]
        elif agg == "mean":
            parent_score = float(np.mean([t[0] for t in top_lst]))
        else:
            raise ValueError("agg must be 'max' or 'mean'")

        parent_rows.append({"parent_path": parent, "parent_score": parent_score, "matches": top_lst})

    parent_rows.sort(key=lambda r: r["parent_score"], reverse=True)
    return parent_rows


# -----------------------------
# UI
# -----------------------------
st.title("Query → Parent (baseline vs segmentation)")

colL, colR = st.columns([1, 2], gap="large")

with colL:
    mode = st.radio("Mode scoring", ["dino", "clip", "fusion"], index=2, horizontal=True)
    alpha = st.slider("alpha (poids DINO en fusion)", 0.0, 1.0, 0.7, 0.05)

    # Baseline full-image dumps
    baseline_dino_dump = st.text_input("Dump baseline DINO (full images)", value="dump_dino_scenes_full.pk1")
    baseline_clip_dump = st.text_input("Dump baseline CLIP (full images)", value="dump_clip_scenes_full.pk1")
    topk_baseline = st.slider("Top-K baseline", 1, 30, 10)

    # Object/crop dumps (DB)
    obj_dino_dump = st.text_input("Dump objets DINO (payload)", value="dump_obj_segnet_scenes.pk1")
    obj_clip_dump = st.text_input("Dump objets CLIP (payload)", value="dump_obj_clip_segnet_scenes.pk1")

    topk_parents = st.slider("Top-K parents", 1, 30, 10)
    topn_crops = st.slider("Top-N crops DB (candidats)", 10, 5000, 400, step=10)
    agg_parent = st.radio("Agrégation par parent", ["max", "mean"], index=0, horizontal=True)
    top_per_parent = st.slider("Nb crops affichés / parent", 1, 5, 1)
    show_bbox = st.checkbox("Afficher bbox sur l'image parent", value=True)

    # Query mode
    query_kind = st.radio("Type requête", ["Crop (déjà segmenté)", "Image complète (segmenter)"], index=0)
    seg_method = st.selectbox("Méthode segmentation requête", ["maskrcnn", "sam_auto", "segnet", "unet"])

    # seg params
    sam_ckpt = st.text_input("SAM checkpoint (si sam_auto)", value="sam_vit_b_01ec64.pth")
    sam_type = st.selectbox("SAM model_type", ["vit_b", "vit_l", "vit_h"], index=0)

    segnet_ckpt = st.text_input("Checkpoint SegNet (si segnet)", value="checkpoints/segnet_objectness_scenes.pt")
    unet_ckpt = st.text_input("Checkpoint UNet (si unet)", value="checkpoints/unet_objectness_scenes.pt")

    uploaded = st.file_uploader("Upload image requête (crop ou full)", type=["jpg", "jpeg", "png"])

with colR:
    if uploaded is None:
        st.info("Upload une image pour lancer.")
        st.stop()

    q_img_full = Image.open(uploaded)
    st.subheader("Image requête (originale)")
    st.image(q_img_full, use_container_width=True)

    # Load models
    d_dev, d_proc, d_model = load_dino()
    if mode in ("clip", "fusion"):
        c_dev, c_proc, c_model = load_clip()

    # Full-image query embeddings (baseline uses this)
    q_dino_full = dino_embed(q_img_full, d_dev, d_proc, d_model)
    q_clip_full = clip_embed(q_img_full, c_dev, c_proc, c_model) if mode in ("clip", "fusion") else None

    # Build query crops (either 1 crop = user input OR segment)
    query_crops = []
    if query_kind.startswith("Crop"):
        query_crops = [{"crop": q_img_full, "bbox": None, "score": None}]
    else:
        try:
            if seg_method == "maskrcnn":
                query_crops = segment_query_maskrcnn(q_img_full)
            elif seg_method == "sam_auto":
                query_crops = segment_query_sam_auto(q_img_full, sam_ckpt=sam_ckpt, model_type=sam_type)
            elif seg_method == "segnet":
                query_crops = segment_query_unet_or_segnet(q_img_full, "segnet", segnet_ckpt)
            elif seg_method == "unet":
                query_crops = segment_query_unet_or_segnet(q_img_full, "unet", unet_ckpt)
        except Exception as e:
            st.error(f"Segmentation requête échouée: {e}")
            st.stop()

    if len(query_crops) == 0:
        st.warning("Aucun crop trouvé sur la requête (segmentation). Essaye d'ajuster les seuils / méthode.")
        st.stop()

    st.subheader(f"Crops requête ({len(query_crops)})")
    colsQ = st.columns(5)
    for i, rec in enumerate(query_crops[:20]):
        colsQ[i % 5].image(to_rgb_flat(rec["crop"]), caption=f"qcrop#{i+1}", use_container_width=True)

    # Query-crop embeddings (for segmented retrieval)
    q_dino_crops = np.stack([dino_embed(r["crop"], d_dev, d_proc, d_model) for r in query_crops], axis=0)  # (Q,D)
    q_clip_crops = None
    if mode in ("clip", "fusion"):
        q_clip_crops = np.stack([clip_embed(r["crop"], c_dev, c_proc, c_model) for r in query_crops], axis=0)

    # Load DB crop dumps
    if not os.path.exists(obj_dino_dump):
        st.error(f"Object dump DINO introuvable: {obj_dino_dump}")
        st.stop()
    db_crop_paths_d, db_crop_embs_d, db_meta = load_vec_dump(obj_dino_dump)
    if db_meta is None:
        st.error("Ton dump objets DINO doit contenir meta (parent_path/bbox). Rebuild via build_object_dump.py.")
        st.stop()

    # Optional DB CLIP crop dump
    db_crop_paths = db_crop_paths_d
    db_scores = None

    if mode == "dino":
        sims = db_crop_embs_d @ q_dino_crops.T  # (M,Q)
        db_scores = sims.max(axis=1)            # best match across query crops

    else:
        if not os.path.exists(obj_clip_dump):
            st.error(f"Object dump CLIP introuvable: {obj_clip_dump} (mode {mode})")
            st.stop()
        db_crop_paths_c, db_crop_embs_c, _ = load_vec_dump(obj_clip_dump)

        # align on common crop paths (canon)
        map_c = {p: i for i, p in enumerate(db_crop_paths_c)}
        common = [p for p in db_crop_paths_d if p in map_c]
        if len(common) == 0:
            st.error("Aucun crop commun entre dump objets DINO et dump objets CLIP. Problème de chemins.")
            st.stop()

        idx_d = [i for i, p in enumerate(db_crop_paths_d) if p in map_c]
        idx_c = [map_c[p] for p in common]

        db_crop_paths = common
        db_d = db_crop_embs_d[idx_d]
        db_c = db_crop_embs_c[idx_c]

        sims_d = db_d @ q_dino_crops.T
        sims_c = db_c @ q_clip_crops.T

        best_d = sims_d.max(axis=1)
        best_c = sims_c.max(axis=1)

        if mode == "clip":
            db_scores = best_c
        else:
            db_scores = alpha * best_d + (1.0 - alpha) * best_c

    # Baseline full-image retrieval
    baseline_results = []
    if mode == "dino":
        if os.path.exists(baseline_dino_dump):
            full_paths, full_embs, _ = load_vec_dump(baseline_dino_dump)
            sims_full = full_embs @ q_dino_full
            topb = np.argsort(-sims_full)[: int(topk_baseline)]
            baseline_results = [(full_paths[i], float(sims_full[i])) for i in topb]
        else:
            st.warning(f"Baseline DINO introuvable: {baseline_dino_dump}")

    elif mode == "clip":
        if os.path.exists(baseline_clip_dump):
            full_paths, full_embs, _ = load_vec_dump(baseline_clip_dump)
            sims_full = full_embs @ q_clip_full
            topb = np.argsort(-sims_full)[: int(topk_baseline)]
            baseline_results = [(full_paths[i], float(sims_full[i])) for i in topb]
        else:
            st.warning(f"Baseline CLIP introuvable: {baseline_clip_dump}")

    else:
        if not (os.path.exists(baseline_dino_dump) and os.path.exists(baseline_clip_dump)):
            st.warning("Fusion baseline: il faut les deux dumps (DINO+CLIP).")
        else:
            d_paths, d_embs, _ = load_vec_dump(baseline_dino_dump)
            c_paths, c_embs, _ = load_vec_dump(baseline_clip_dump)
            c_map = {p: i for i, p in enumerate(c_paths)}
            common = [p for p in d_paths if p in c_map]
            if len(common) > 0:
                d_map = {p: i for i, p in enumerate(d_paths)}
                d_idx = [d_map[p] for p in common]
                c_idx = [c_map[p] for p in common]
                sims_full = alpha * (d_embs[d_idx] @ q_dino_full) + (1.0 - alpha) * (c_embs[c_idx] @ q_clip_full)
                topb = np.argsort(-sims_full)[: int(topk_baseline)]
                baseline_results = [(common[i], float(sims_full[i])) for i in topb]

    # Top DB crops → parents
    topn = min(int(topn_crops), len(db_crop_paths))
    idx = np.argsort(-db_scores)[:topn]
    top_db_crop_paths = [db_crop_paths[i] for i in idx]
    top_db_crop_scores = db_scores[idx]

    tab1, tab2 = st.tabs(["Sans segmentation (baseline)", "Avec segmentation (crop→parent)"])

    with tab1:
        st.subheader("Résultats baseline (full images)")
        colsB = st.columns(5)
        for j, (p, s) in enumerate(baseline_results):
            c = colsB[j % 5]
            if os.path.exists(p):
                c.image(Image.open(p).convert("RGB"), caption=f"{j+1}. score={s:.3f}\n{Path(p).name}", use_container_width=True)
            else:
                c.write(f"{j+1}. {Path(p).name} (missing)")

    with tab2:
        parents_ranked = rank_parents_from_crop_scores(
            top_db_crop_paths, top_db_crop_scores, db_meta, agg=agg_parent, top_per_parent=int(top_per_parent)
        )[: int(topk_parents)]

        st.subheader("Résultats (images parent)")
        cols = st.columns(2)
        for j, row in enumerate(parents_ranked):
            parent_path = row["parent_path"]
            parent_score = row["parent_score"]
            matches = row["matches"]

            c = cols[j % 2]
            if not os.path.exists(parent_path):
                c.warning(f"Parent introuvable: {parent_path}")
                continue

            parent_img = Image.open(parent_path).convert("RGB")
            best_score, best_crop_path, best_bbox = matches[0]

            if show_bbox and best_bbox is not None:
                try:
                    parent_disp = draw_bbox(parent_img, best_bbox, width=6)
                except Exception:
                    parent_disp = parent_img
            else:
                parent_disp = parent_img

            c.image(parent_disp, caption=f"{j+1}. score_parent={parent_score:.3f}\n{Path(parent_path).name}", use_container_width=True)

            with c.expander("Détails du match"):
                c.write(f"Parent: {parent_path}")
                for k_idx, (s, cp, bb) in enumerate(matches, start=1):
                    c.write(f"- match#{k_idx} score={s:.3f} crop={Path(cp).name}")
                    if os.path.exists(cp):
                        c.image(to_rgb_flat(Image.open(cp)), caption=f"crop match#{k_idx}", use_container_width=True)