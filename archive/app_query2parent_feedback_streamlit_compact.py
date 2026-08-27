# app_query2parent_feedback_streamlit.py

import os
import io
import hashlib
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


st.set_page_config(
    page_title="Segmented CBIR with Feedback",
    layout="wide"
)


# ============================================================
# General utilities
# ============================================================

def canon(p: str) -> str:
    return os.path.normcase(os.path.normpath(str(p)))


def normalize(v: np.ndarray) -> np.ndarray:
    return v / (np.linalg.norm(v) + 1e-12)


def to_rgb_flat(img: Image.Image, bg=(127, 127, 127)) -> Image.Image:
    """
    Converts any PIL image to RGB.
    If the image contains an alpha channel, it is flattened on a neutral grey background.
    """
    if img.mode in ("RGBA", "LA") or (img.mode == "P" and "transparency" in img.info):
        img = img.convert("RGBA")
        bg_img = Image.new("RGBA", img.size, (*bg, 255))
        img = Image.alpha_composite(bg_img, img).convert("RGB")
        return img

    return img.convert("RGB")


def apply_mask_rgba(crop_rgb: np.ndarray, mask: np.ndarray) -> Image.Image:
    """
    Creates a RGBA crop where the mask is used as alpha channel.
    """
    alpha = (mask.astype(np.uint8) * 255)
    rgba = np.dstack([crop_rgb, alpha])
    return Image.fromarray(rgba, mode="RGBA")


def draw_bbox(parent_img: Image.Image, bbox_xyxy, width=6):
    """
    Draws a red bounding box on the parent image.
    """
    img = parent_img.copy().convert("RGB")
    draw = ImageDraw.Draw(img)

    x1, y1, x2, y2 = [int(v) for v in bbox_xyxy]

    for w in range(width):
        draw.rectangle([x1 - w, y1 - w, x2 + w, y2 + w], outline=(255, 0, 0))

    return img



def make_thumbnail(img: Image.Image, size=(360, 260), bg=(245, 245, 245)) -> Image.Image:
    """
    Builds a fixed-size RGB thumbnail with padding.

    Streamlit otherwise displays images with heterogeneous heights, which makes
    top-K grids difficult to read. This helper preserves the aspect ratio and
    centers the image in a fixed canvas.
    """
    img = to_rgb_flat(img)
    thumb = img.copy()
    thumb.thumbnail(size, Image.LANCZOS)

    canvas = Image.new("RGB", size, bg)
    x = (size[0] - thumb.size[0]) // 2
    y = (size[1] - thumb.size[1]) // 2
    canvas.paste(thumb, (x, y))

    return canvas


def uploaded_file_key(uploaded, query_kind: str, seg_method: str, mode: str) -> str:
    """
    Stable key used to decide when the feedback state must be reset.
    """
    raw = uploaded.getvalue()
    h = hashlib.md5(raw).hexdigest()
    return f"{h}_{query_kind}_{seg_method}_{mode}"


# ============================================================
# Dump loaders
# ============================================================

@st.cache_data
def load_vec_dump(dump_path: str):
    """
    Accepts either:
      1. dict[path -> embedding]
      2. payload {"embeddings": dict[path -> embedding], "meta": dict}

    Returns:
      paths: list[str]
      embs: np.ndarray of shape (N, D), L2-normalized
      meta: dict or None
    """
    with open(dump_path, "rb") as f:
        obj = pickle.load(f)

    meta = None

    if isinstance(obj, dict) and "embeddings" in obj:
        emb_dict = obj["embeddings"]
        meta = obj.get("meta", None)
    else:
        emb_dict = obj

    emb_dict = {canon(k): v for k, v in emb_dict.items()}

    paths = list(emb_dict.keys())
    embs = np.stack([emb_dict[p] for p in paths], axis=0).astype(np.float32)
    embs = embs / (np.linalg.norm(embs, axis=1, keepdims=True) + 1e-12)

    meta_norm = None
    if isinstance(meta, dict):
        meta_norm = {}
        for k, v in meta.items():
            kk = canon(k)
            vv = dict(v)

            if "parent_path" in vv and vv["parent_path"] is not None:
                vv["parent_path"] = canon(vv["parent_path"])

            if "crop_path" in vv and vv["crop_path"] is not None:
                vv["crop_path"] = canon(vv["crop_path"])

            meta_norm[kk] = vv

    return paths, embs, meta_norm


# ============================================================
# DINOv2 and CLIP models
# ============================================================

@st.cache_resource
def load_dino():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    processor = AutoImageProcessor.from_pretrained("facebook/dinov2-base")
    model = AutoModel.from_pretrained("facebook/dinov2-base").to(device)
    model.eval()
    return device, processor, model


@st.cache_resource
def load_clip():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
    model.eval()
    return device, processor, model


def dino_embed(img: Image.Image, device, processor, model) -> np.ndarray:
    img = to_rgb_flat(img)
    inputs = processor(images=img, return_tensors="pt")
    pixel_values = inputs["pixel_values"].to(device)

    with torch.no_grad():
        outputs = model(pixel_values)
        emb = outputs.last_hidden_state.mean(dim=1).squeeze(0)
        emb = emb.detach().cpu().numpy().astype(np.float32)

    return normalize(emb)


def clip_embed(img: Image.Image, device, processor, model) -> np.ndarray:
    img = to_rgb_flat(img)
    inputs = processor(images=img, return_tensors="pt")
    pixel_values = inputs["pixel_values"].to(device)

    with torch.no_grad():
        emb = model.get_image_features(pixel_values=pixel_values).squeeze(0)
        emb = emb.detach().cpu().numpy().astype(np.float32)

    return normalize(emb)


# ============================================================
# Mask R-CNN model for query-time segmentation
# ============================================================

@st.cache_resource
def load_maskrcnn():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    weights = MaskRCNN_ResNet50_FPN_V2_Weights.DEFAULT
    model = maskrcnn_resnet50_fpn_v2(weights=weights).to(device)
    model.eval()
    preprocess = weights.transforms()
    return device, model, preprocess


def segment_query_maskrcnn(
    img: Image.Image,
    score_thr=0.7,
    max_objs=5,
    min_area=100,
    pad=5,
):
    """
    Segment query image using Mask R-CNN.
    Returns a list of dicts:
      {"crop": PIL.Image, "bbox": (x1,y1,x2,y2), "score": float}
    """
    device, model, preprocess = load_maskrcnn()

    img_rgb = to_rgb_flat(img)
    x = preprocess(img_rgb).to(device)

    with torch.no_grad():
        pred = model([x])[0]

    scores = pred["scores"].detach().cpu().numpy()
    masks = pred["masks"].detach().cpu().numpy()

    keep = np.where(scores >= score_thr)[0][:max_objs]

    if len(keep) == 0:
        return []

    img_np = np.array(img_rgb)
    H, W = img_np.shape[:2]

    out = []

    for j in keep:
        m = masks[j, 0] > 0.5

        if int(m.sum()) < min_area:
            continue

        ys, xs = np.where(m)

        y1, y2 = int(ys.min()), int(ys.max())
        x1, x2 = int(xs.min()), int(xs.max())

        x1 = max(0, x1 - pad)
        y1 = max(0, y1 - pad)
        x2 = min(W - 1, x2 + pad)
        y2 = min(H - 1, y2 + pad)

        crop = img_np[y1:y2 + 1, x1:x2 + 1]
        crop_mask = m[y1:y2 + 1, x1:x2 + 1]

        crop_rgba = apply_mask_rgba(crop, crop_mask)

        out.append({
            "crop": crop_rgba,
            "bbox": (x1, y1, x2, y2),
            "score": float(scores[j]),
            "method": "maskrcnn",
        })

    return out


# ============================================================
# SAM automatic segmentation
# ============================================================

@st.cache_resource
def load_sam_model(sam_checkpoint: str, model_type: str):
    from segment_anything import sam_model_registry

    device = "cuda" if torch.cuda.is_available() else "cpu"

    sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
    sam.to(device=device)

    return device, sam


def segment_query_sam_auto(
    img: Image.Image,
    sam_checkpoint: str,
    model_type="vit_b",
    points_per_side=16,
    pred_iou_thresh=0.86,
    stability_score_thresh=0.92,
    crop_n_layers=0,
    max_masks=15,
    min_area=400,
    max_area_ratio=0.65,
    pad=5,
):
    """
    Segment query image using SAM automatic mask generation.
    """
    try:
        from segment_anything import SamAutomaticMaskGenerator
    except Exception as e:
        raise RuntimeError(f"segment-anything is not available: {e}")

    device, sam = load_sam_model(sam_checkpoint, model_type)

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

        if (area / float(H * W)) > max_area_ratio:
            continue

        x, y, w, h = ann["bbox"]

        x1 = int(x)
        y1 = int(y)
        x2 = int(x + w - 1)
        y2 = int(y + h - 1)

        x1 = max(0, x1 - pad)
        y1 = max(0, y1 - pad)
        x2 = min(W - 1, x2 + pad)
        y2 = min(H - 1, y2 + pad)

        crop = img_np[y1:y2 + 1, x1:x2 + 1]
        crop_mask = m[y1:y2 + 1, x1:x2 + 1]

        crop_rgba = apply_mask_rgba(crop, crop_mask)

        out.append({
            "crop": crop_rgba,
            "bbox": (x1, y1, x2, y2),
            "score": float(ann.get("predicted_iou", 0.0)),
            "method": "sam_auto",
        })

        kept += 1

    return out


# ============================================================
# SegNet / U-Net definitions
# ============================================================

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
        x1 = self.enc1(x)
        s1 = x1.size()
        x, i1 = self.pool1(x1)

        x2 = self.enc2(x)
        s2 = x2.size()
        x, i2 = self.pool2(x2)

        x3 = self.enc3(x)
        s3 = x3.size()
        x, i3 = self.pool3(x3)

        x4 = self.enc4(x)
        s4 = x4.size()
        x, i4 = self.pool4(x4)

        x5 = self.enc5(x)
        s5 = x5.size()
        x, i5 = self.pool5(x5)

        x = self.unpool5(x, i5, output_size=s5)
        x = self.dec5(x)

        x = self.unpool4(x, i4, output_size=s4)
        x = self.dec4(x)

        x = self.unpool3(x, i3, output_size=s3)
        x = self.dec3(x)

        x = self.unpool2(x, i2, output_size=s2)
        x = self.dec2(x)

        x = self.unpool1(x, i1, output_size=s1)
        x = self.dec1(x)

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

    def forward(self, x):
        return self.net(x)


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

        y4 = self.up4(xm)
        y4 = self.conv4(torch.cat([y4, x4], dim=1))

        y3 = self.up3(y4)
        y3 = self.conv3(torch.cat([y3, x3], dim=1))

        y2 = self.up2(y3)
        y2 = self.conv2(torch.cat([y2, x2], dim=1))

        y1 = self.up1(y2)
        y1 = self.conv1(torch.cat([y1, x1], dim=1))

        return self.out(y1)


@st.cache_resource
def load_segnet_ckpt(ckpt_path: str):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    model = SegNetBinary().to(device)

    ckpt = torch.load(ckpt_path, map_location=device)

    if isinstance(ckpt, dict) and "state_dict" in ckpt:
        state = ckpt["state_dict"]
        size = int(ckpt.get("size", 256))
    else:
        state = ckpt
        size = 256

    model.load_state_dict(state, strict=True)
    model.eval()

    return device, model, size


@st.cache_resource
def load_unet_ckpt(ckpt_path: str):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    model = UNet().to(device)

    ckpt = torch.load(ckpt_path, map_location=device)

    if isinstance(ckpt, dict) and "state_dict" in ckpt:
        state = ckpt["state_dict"]
        size = int(ckpt.get("size", 384))
    else:
        state = ckpt
        size = 384

    model.load_state_dict(state, strict=True)
    model.eval()

    return device, model, size


def cc_bboxes_from_mask(mask: np.ndarray, min_area=300, max_objs=10):
    labeled, n_cc = cc_label(mask)

    if n_cc == 0:
        return []

    slices = find_objects(labeled)

    comps = []

    for cc_id, sl in enumerate(slices, start=1):
        if sl is None:
            continue

        cc_mask = labeled[sl] == cc_id
        area = int(cc_mask.sum())

        comps.append((area, cc_id, sl))

    comps.sort(reverse=True, key=lambda t: t[0])
    comps = comps[:max_objs]

    out = []

    for area, cc_id, sl in comps:
        if area < min_area:
            continue

        ysl, xsl = sl

        y1 = ysl.start
        y2 = ysl.stop - 1
        x1 = xsl.start
        x2 = xsl.stop - 1

        out.append((cc_id, x1, y1, x2, y2))

    return out


def segment_query_unet_or_segnet(
    img: Image.Image,
    method: str,
    ckpt_path: str,
    threshold=0.5,
    input_size=None,
    min_area=300,
    max_objs=10,
    max_area_ratio=0.75,
):
    """
    Segment query image using a trained binary SegNet or U-Net.
    """
    img_rgb = to_rgb_flat(img)
    img_np = np.array(img_rgb)

    H, W = img_np.shape[:2]

    if method == "segnet":
        device, model, default_size = load_segnet_ckpt(ckpt_path)
        size = default_size if input_size is None else input_size
    elif method == "unet":
        device, model, default_size = load_unet_ckpt(ckpt_path)
        size = default_size if input_size is None else input_size
    else:
        raise ValueError(f"Unknown method: {method}")

    x = torch.from_numpy(img_np).permute(2, 0, 1).float() / 255.0
    x = x.unsqueeze(0).to(device)

    x_rs = F.interpolate(
        x,
        size=(size, size),
        mode="bilinear",
        align_corners=False,
    )

    with torch.no_grad():
        logits = model(x_rs)
        prob = torch.sigmoid(logits)

    prob = F.interpolate(
        prob,
        size=(H, W),
        mode="bilinear",
        align_corners=False,
    )

    prob = prob.squeeze(0).squeeze(0)
    mask = prob.detach().cpu().numpy() >= threshold

    area = int(mask.sum())

    if area == 0:
        return []

    if (area / float(H * W)) > max_area_ratio:
        return []

    bboxes = cc_bboxes_from_mask(
        mask,
        min_area=min_area,
        max_objs=max_objs,
    )

    out = []

    for cc_id, x1, y1, x2, y2 in bboxes:
        crop = img_np[y1:y2 + 1, x1:x2 + 1]
        crop_mask = mask[y1:y2 + 1, x1:x2 + 1]

        crop_rgba = apply_mask_rgba(crop, crop_mask)

        out.append({
            "crop": crop_rgba,
            "bbox": (x1, y1, x2, y2),
            "score": None,
            "method": method,
        })

    return out


# ============================================================
# Crop to parent ranking
# ============================================================

def rank_parents_from_crop_scores(
    crop_paths,
    crop_scores,
    meta,
    agg="max",
    top_per_parent=3,
):
    """
    Aggregates crop-level scores into parent-image scores.

    If agg = "max":
        S(parent) = max crop score

    If agg = "mean":
        S(parent) = mean of top-m crop scores
    """
    bucket = defaultdict(list)

    for p, s in zip(crop_paths, crop_scores):
        cp = canon(p)
        rec = meta.get(cp) if meta else None

        if not rec:
            continue

        parent = rec.get("parent_path", None)
        bbox = rec.get("bbox_xyxy", None)

        if not parent:
            continue

        bucket[parent].append((float(s), cp, bbox))

    parent_rows = []

    for parent, lst in bucket.items():
        lst.sort(key=lambda t: t[0], reverse=True)

        top_lst = lst[:top_per_parent]

        if len(top_lst) == 0:
            continue

        if agg == "max":
            parent_score = top_lst[0][0]
        elif agg == "mean":
            parent_score = float(np.mean([t[0] for t in top_lst]))
        else:
            raise ValueError("agg must be 'max' or 'mean'")

        parent_rows.append({
            "parent_path": parent,
            "parent_score": parent_score,
            "matches": top_lst,
        })

    parent_rows.sort(key=lambda r: r["parent_score"], reverse=True)

    return parent_rows


# ============================================================
# Local relevance feedback
# ============================================================

def reset_feedback_state():
    keys = [
        "feedback_query_key",
        "pos_dino_vecs",
        "neg_dino_vecs",
        "pos_clip_vecs",
        "neg_clip_vecs",
        "feedback_iteration",
        "positive_parent_paths",
        "negative_parent_paths",
        "positive_crop_paths",
        "negative_crop_paths",
    ]

    for key in keys:
        if key in st.session_state:
            del st.session_state[key]


def ensure_segmented_feedback_state(
    query_key,
    q_dino_crops,
    q_clip_crops=None,
):
    """
    Initializes local feedback for a new query.

    Initial positive prototypes = query crops.
    Negative prototypes = empty.
    """
    if st.session_state.get("feedback_query_key") != query_key:
        st.session_state.feedback_query_key = query_key

        st.session_state.pos_dino_vecs = [
            v.copy() for v in q_dino_crops
        ]

        st.session_state.neg_dino_vecs = []

        if q_clip_crops is not None:
            st.session_state.pos_clip_vecs = [
                v.copy() for v in q_clip_crops
            ]
        else:
            st.session_state.pos_clip_vecs = []

        st.session_state.neg_clip_vecs = []

        st.session_state.feedback_iteration = 0
        st.session_state.positive_parent_paths = []
        st.session_state.negative_parent_paths = []
        st.session_state.positive_crop_paths = []
        st.session_state.negative_crop_paths = []


def score_crops_with_local_feedback(
    mode,
    alpha,
    db_dino,
    db_clip,
    neg_weight=0.4,
):
    """
    Scores every database crop using local multi-prototype feedback.

    For each database crop c:

        score(c) = max_p cos(c,p) - lambda max_n cos(c,n)

    where p are positive local prototypes and n are negative local prototypes.

    In fusion mode, the score is computed separately in DINO and CLIP spaces,
    then fused with alpha.
    """
    scores_dino = None
    scores_clip = None

    if mode in ("dino", "fusion"):
        pos = np.stack(st.session_state.pos_dino_vecs, axis=0).astype(np.float32)
        pos_scores = db_dino @ pos.T
        scores_dino = pos_scores.max(axis=1)

        if len(st.session_state.neg_dino_vecs) > 0:
            neg = np.stack(st.session_state.neg_dino_vecs, axis=0).astype(np.float32)
            neg_scores = db_dino @ neg.T
            scores_dino = scores_dino - neg_weight * neg_scores.max(axis=1)

    if mode in ("clip", "fusion"):
        if db_clip is None:
            raise RuntimeError("CLIP mode requested but db_clip is None.")

        pos = np.stack(st.session_state.pos_clip_vecs, axis=0).astype(np.float32)
        pos_scores = db_clip @ pos.T
        scores_clip = pos_scores.max(axis=1)

        if len(st.session_state.neg_clip_vecs) > 0:
            neg = np.stack(st.session_state.neg_clip_vecs, axis=0).astype(np.float32)
            neg_scores = db_clip @ neg.T
            scores_clip = scores_clip - neg_weight * neg_scores.max(axis=1)

    if mode == "dino":
        return scores_dino

    if mode == "clip":
        return scores_clip

    return alpha * scores_dino + (1.0 - alpha) * scores_clip


def add_feedback_from_selected_rows(
    selected_positive_rows,
    selected_negative_rows,
    crop_to_idx,
    db_dino,
    db_clip=None,
):
    """
    Adds the crops responsible for selected parent results to the feedback prototypes.

    If a parent result is marked relevant:
        its best-matching crop becomes a positive local prototype.

    If a parent result is marked non-relevant:
        its best-matching crop becomes a negative local prototype.
    """
    for row in selected_positive_rows:
        parent_path = row["parent_path"]
        _, crop_path, _ = row["matches"][0]

        cp = canon(crop_path)

        if cp not in crop_to_idx:
            continue

        idx = crop_to_idx[cp]

        st.session_state.pos_dino_vecs.append(db_dino[idx].copy())

        if db_clip is not None:
            st.session_state.pos_clip_vecs.append(db_clip[idx].copy())

        st.session_state.positive_parent_paths.append(parent_path)
        st.session_state.positive_crop_paths.append(cp)

    for row in selected_negative_rows:
        parent_path = row["parent_path"]
        _, crop_path, _ = row["matches"][0]

        cp = canon(crop_path)

        if cp not in crop_to_idx:
            continue

        idx = crop_to_idx[cp]

        st.session_state.neg_dino_vecs.append(db_dino[idx].copy())

        if db_clip is not None:
            st.session_state.neg_clip_vecs.append(db_clip[idx].copy())

        st.session_state.negative_parent_paths.append(parent_path)
        st.session_state.negative_crop_paths.append(cp)

    st.session_state.feedback_iteration += 1


# ============================================================
# Streamlit UI
# ============================================================

st.title("Segmented CBIR with interactive local feedback")

left, right = st.columns([0.82, 2.18], gap="large")

with left:
    st.header("Recherche")

    uploaded = st.file_uploader(
        "Image requête",
        type=["jpg", "jpeg", "png"],
    )

    query_kind = st.radio(
        "Type de requête",
        ["Crop déjà segmenté", "Image complète à segmenter"],
        index=0,
    )

    mode = st.radio(
        "Scoring",
        ["dino", "clip", "fusion"],
        index=2,
        horizontal=True,
    )

    if mode == "fusion":
        alpha = st.slider(
            "Poids DINO dans la fusion",
            min_value=0.0,
            max_value=1.0,
            value=0.7,
            step=0.05,
        )
    elif mode == "dino":
        alpha = 1.0
    else:
        alpha = 0.0

    if query_kind == "Image complète à segmenter":
        seg_method = st.selectbox(
            "Segmentation de la requête",
            ["maskrcnn", "sam_auto", "segnet", "unet"],
            index=0,
        )
    else:
        seg_method = "manual_crop"

    with st.expander("Chemins des dumps", expanded=False):
        baseline_dino_dump = st.text_input(
            "Dump baseline DINO",
            value="dump_dino_scenes_full.pk1",
        )

        baseline_clip_dump = st.text_input(
            "Dump baseline CLIP",
            value="dump_clip_scenes_full.pk1",
        )

        obj_dino_dump = st.text_input(
            "Dump objets DINO",
            value="dump_obj_segnet_scenes.pk1",
        )

        obj_clip_dump = st.text_input(
            "Dump objets CLIP",
            value="dump_obj_clip_segnet_scenes.pk1",
        )

    with st.expander("Paramètres avancés de recherche", expanded=False):
        topk_baseline = st.slider(
            "Top-K baseline",
            min_value=1,
            max_value=30,
            value=10,
        )

        topk_parents = st.slider(
            "Top-K résultats segmentés",
            min_value=1,
            max_value=30,
            value=10,
        )

        topn_crops = st.slider(
            "Top-N crops candidats",
            min_value=10,
            max_value=5000,
            value=400,
            step=10,
        )

        agg_parent = st.radio(
            "Agrégation parent",
            ["max", "mean"],
            index=0,
            horizontal=True,
        )

        top_per_parent = st.slider(
            "Crops conservés par parent",
            min_value=1,
            max_value=5,
            value=1,
        )

        show_bbox = st.checkbox(
            "Afficher la bbox",
            value=True,
        )

    with st.expander("Feedback local", expanded=True):
        enable_feedback = st.checkbox(
            "Activer le feedback crop→parent",
            value=True,
        )

        neg_weight = st.slider(
            "Poids des exemples non pertinents",
            min_value=0.0,
            max_value=2.0,
            value=0.4,
            step=0.1,
        )

        hide_annotated_parents = st.checkbox(
            "Masquer les parents déjà annotés",
            value=False,
        )

        if st.button("Réinitialiser le feedback"):
            reset_feedback_state()
            st.rerun()

    # Defaults for query segmentation. They are hidden unless the query is a full image.
    sam_ckpt = "sam_vit_b_01ec64.pth"
    sam_type = "vit_b"
    segnet_ckpt = "checkpoints/segnet_objectness_scenes.pt"
    unet_ckpt = "checkpoints/unet_objectness_scenes.pt"

    if query_kind == "Image complète à segmenter":
        with st.expander("Paramètres segmentation", expanded=False):
            if seg_method == "sam_auto":
                sam_ckpt = st.text_input(
                    "SAM checkpoint",
                    value=sam_ckpt,
                )

                sam_type = st.selectbox(
                    "SAM model type",
                    ["vit_b", "vit_l", "vit_h"],
                    index=0,
                )

            elif seg_method == "segnet":
                segnet_ckpt = st.text_input(
                    "Checkpoint SegNet",
                    value=segnet_ckpt,
                )

            elif seg_method == "unet":
                unet_ckpt = st.text_input(
                    "Checkpoint U-Net",
                    value=unet_ckpt,
                )

            else:
                st.caption("Mask R-CNN utilise les poids COCO pré-entraînés.")
with right:
    if uploaded is None:
        st.info("Charge une image requête pour lancer la recherche.")
        st.stop()

    # ------------------------------------------------------------
    # Load query image
    # ------------------------------------------------------------
    query_bytes = uploaded.getvalue()
    q_img_full = Image.open(io.BytesIO(query_bytes))
    q_img_full = to_rgb_flat(q_img_full)

    st.subheader("Image requête originale")
    st.image(q_img_full, use_container_width=True)

    base_query_key = uploaded_file_key(
        uploaded,
        query_kind=query_kind,
        seg_method=seg_method,
        mode=mode,
    )

    # ------------------------------------------------------------
    # Load embedding models
    # ------------------------------------------------------------
    dino_device, dino_processor, dino_model = load_dino()

    if mode in ("clip", "fusion"):
        clip_device, clip_processor, clip_model = load_clip()
    else:
        clip_device, clip_processor, clip_model = None, None, None

    # ------------------------------------------------------------
    # Full-image query embeddings for baseline
    # ------------------------------------------------------------
    q_dino_full = dino_embed(
        q_img_full,
        dino_device,
        dino_processor,
        dino_model,
    )

    q_clip_full = None

    if mode in ("clip", "fusion"):
        q_clip_full = clip_embed(
            q_img_full,
            clip_device,
            clip_processor,
            clip_model,
        )

    # ------------------------------------------------------------
    # Query crops
    # ------------------------------------------------------------
    if query_kind == "Crop déjà segmenté":
        query_crops = [{
            "crop": q_img_full,
            "bbox": None,
            "score": None,
            "method": "manual_crop",
        }]
        selected_query_crop_idx = "manual"

        st.subheader("Crop requête utilisé")
        st.image(
            make_thumbnail(q_img_full, size=(420, 300)),
            caption="crop manuel",
            use_container_width=False,
        )

    else:
        try:
            if seg_method == "maskrcnn":
                detected_query_crops = segment_query_maskrcnn(q_img_full)

            elif seg_method == "sam_auto":
                detected_query_crops = segment_query_sam_auto(
                    q_img_full,
                    sam_checkpoint=sam_ckpt,
                    model_type=sam_type,
                )

            elif seg_method == "segnet":
                detected_query_crops = segment_query_unet_or_segnet(
                    q_img_full,
                    method="segnet",
                    ckpt_path=segnet_ckpt,
                )

            elif seg_method == "unet":
                detected_query_crops = segment_query_unet_or_segnet(
                    q_img_full,
                    method="unet",
                    ckpt_path=unet_ckpt,
                )

            else:
                raise ValueError(seg_method)

        except Exception as e:
            st.error(f"Segmentation de la requête échouée : {e}")
            st.stop()

        if len(detected_query_crops) == 0:
            st.warning("Aucun crop trouvé dans l'image requête.")
            st.stop()

        st.subheader(f"Crops détectés dans la requête ({len(detected_query_crops)})")

        q_cols = st.columns(5)
        for i, rec in enumerate(detected_query_crops[:20]):
            q_cols[i % 5].image(
                make_thumbnail(rec["crop"], size=(220, 170)),
                caption=f"qcrop #{i + 1}",
                use_container_width=True,
            )

        choice_labels = ["Tous les crops"] + [
            f"qcrop #{i + 1}"
            for i in range(len(detected_query_crops))
        ]

        selected_crop_choice = st.selectbox(
            "Crop utilisé pour la recherche segmentée",
            options=choice_labels,
            index=0,
            help=(
                "Choisis un crop précis si tu veux retrouver principalement ce détail. "
                "Garde 'Tous les crops' pour utiliser tous les segments détectés."
            ),
        )

        if selected_crop_choice == "Tous les crops":
            selected_query_crop_idx = "all"
            query_crops = detected_query_crops
            st.caption(
                "Recherche segmentée basée sur tous les crops détectés. "
                "Le score d'un crop de la base correspond au meilleur match avec les crops de requête."
            )
        else:
            selected_query_crop_idx = int(selected_crop_choice.split("#")[1]) - 1
            query_crops = [detected_query_crops[selected_query_crop_idx]]
            st.success(
                f"Recherche segmentée limitée au crop #{selected_query_crop_idx + 1}. "
                "La baseline full-image reste calculée sur l'image complète."
            )

            st.image(
                make_thumbnail(query_crops[0]["crop"], size=(420, 300)),
                caption=f"Crop actif #{selected_query_crop_idx + 1}",
                use_container_width=False,
            )

    query_key = f"{base_query_key}_selected_query_crop={selected_query_crop_idx}"
    # ------------------------------------------------------------
    # Query crop embeddings
    # ------------------------------------------------------------
    q_dino_crops = np.stack(
        [
            dino_embed(
                rec["crop"],
                dino_device,
                dino_processor,
                dino_model,
            )
            for rec in query_crops
        ],
        axis=0,
    ).astype(np.float32)

    q_clip_crops = None

    if mode in ("clip", "fusion"):
        q_clip_crops = np.stack(
            [
                clip_embed(
                    rec["crop"],
                    clip_device,
                    clip_processor,
                    clip_model,
                )
                for rec in query_crops
            ],
            axis=0,
        ).astype(np.float32)

    # ------------------------------------------------------------
    # Load database crop dumps
    # ------------------------------------------------------------
    if not os.path.exists(obj_dino_dump):
        st.error(f"Dump objets DINO introuvable : {obj_dino_dump}")
        st.stop()

    db_crop_paths_dino, db_crop_embs_dino, db_meta = load_vec_dump(obj_dino_dump)

    if db_meta is None:
        st.error(
            "Le dump objets DINO doit contenir un champ meta avec parent_path et bbox_xyxy."
        )
        st.stop()

    db_crop_paths = db_crop_paths_dino
    db_dino = db_crop_embs_dino
    db_clip = None

    if mode in ("clip", "fusion"):
        if not os.path.exists(obj_clip_dump):
            st.error(f"Dump objets CLIP introuvable : {obj_clip_dump}")
            st.stop()

        db_crop_paths_clip, db_crop_embs_clip, _ = load_vec_dump(obj_clip_dump)

        map_d = {p: i for i, p in enumerate(db_crop_paths_dino)}
        map_c = {p: i for i, p in enumerate(db_crop_paths_clip)}

        common = [p for p in db_crop_paths_dino if p in map_c]

        if len(common) == 0:
            st.error("Aucun crop commun entre les dumps DINO et CLIP.")
            st.stop()

        idx_d = [map_d[p] for p in common]
        idx_c = [map_c[p] for p in common]

        db_crop_paths = common
        db_dino = db_crop_embs_dino[idx_d]
        db_clip = db_crop_embs_clip[idx_c]

    crop_to_idx = {
        canon(p): i
        for i, p in enumerate(db_crop_paths)
    }

    # ------------------------------------------------------------
    # Feedback initialization
    # ------------------------------------------------------------
    if enable_feedback:
        ensure_segmented_feedback_state(
            query_key=query_key,
            q_dino_crops=q_dino_crops,
            q_clip_crops=q_clip_crops if mode in ("clip", "fusion") else None,
        )

    # ------------------------------------------------------------
    # Compute crop scores
    # ------------------------------------------------------------
    if enable_feedback:
        db_scores = score_crops_with_local_feedback(
            mode=mode,
            alpha=alpha,
            db_dino=db_dino,
            db_clip=db_clip,
            neg_weight=neg_weight,
        )

    else:
        if mode == "dino":
            sims = db_dino @ q_dino_crops.T
            db_scores = sims.max(axis=1)

        elif mode == "clip":
            sims = db_clip @ q_clip_crops.T
            db_scores = sims.max(axis=1)

        else:
            sims_d = db_dino @ q_dino_crops.T
            sims_c = db_clip @ q_clip_crops.T

            best_d = sims_d.max(axis=1)
            best_c = sims_c.max(axis=1)

            db_scores = alpha * best_d + (1.0 - alpha) * best_c

    # ------------------------------------------------------------
    # Full-image baseline retrieval
    # ------------------------------------------------------------
    baseline_results = []

    if mode == "dino":
        if os.path.exists(baseline_dino_dump):
            full_paths, full_embs, _ = load_vec_dump(baseline_dino_dump)
            sims_full = full_embs @ q_dino_full
            topb = np.argsort(-sims_full)[:int(topk_baseline)]
            baseline_results = [
                (full_paths[i], float(sims_full[i]))
                for i in topb
            ]
        else:
            st.warning(f"Baseline DINO introuvable : {baseline_dino_dump}")

    elif mode == "clip":
        if os.path.exists(baseline_clip_dump):
            full_paths, full_embs, _ = load_vec_dump(baseline_clip_dump)
            sims_full = full_embs @ q_clip_full
            topb = np.argsort(-sims_full)[:int(topk_baseline)]
            baseline_results = [
                (full_paths[i], float(sims_full[i]))
                for i in topb
            ]
        else:
            st.warning(f"Baseline CLIP introuvable : {baseline_clip_dump}")

    else:
        if not (os.path.exists(baseline_dino_dump) and os.path.exists(baseline_clip_dump)):
            st.warning("Fusion baseline : il faut les deux dumps full-image DINO et CLIP.")
        else:
            d_paths, d_embs, _ = load_vec_dump(baseline_dino_dump)
            c_paths, c_embs, _ = load_vec_dump(baseline_clip_dump)

            c_map = {p: i for i, p in enumerate(c_paths)}
            common_full = [p for p in d_paths if p in c_map]

            if len(common_full) > 0:
                d_map = {p: i for i, p in enumerate(d_paths)}

                d_idx = [d_map[p] for p in common_full]
                c_idx = [c_map[p] for p in common_full]

                sims_full = (
                    alpha * (d_embs[d_idx] @ q_dino_full)
                    + (1.0 - alpha) * (c_embs[c_idx] @ q_clip_full)
                )

                topb = np.argsort(-sims_full)[:int(topk_baseline)]

                baseline_results = [
                    (common_full[i], float(sims_full[i]))
                    for i in topb
                ]

    # ------------------------------------------------------------
    # Top crops -> parent ranking
    # ------------------------------------------------------------
    topn = min(int(topn_crops), len(db_crop_paths))

    top_crop_idx = np.argsort(-db_scores)[:topn]

    top_db_crop_paths = [
        db_crop_paths[i]
        for i in top_crop_idx
    ]

    top_db_crop_scores = db_scores[top_crop_idx]

    all_parent_rows = rank_parents_from_crop_scores(
        top_db_crop_paths,
        top_db_crop_scores,
        db_meta,
        agg=agg_parent,
        top_per_parent=int(top_per_parent),
    )

    if enable_feedback and hide_annotated_parents:
        annotated = set(st.session_state.positive_parent_paths)
        annotated.update(st.session_state.negative_parent_paths)

        all_parent_rows = [
            row for row in all_parent_rows
            if row["parent_path"] not in annotated
        ]

    parents_ranked = all_parent_rows[:int(topk_parents)]

    # ------------------------------------------------------------
    # Tabs display
    # ------------------------------------------------------------
    tab1, tab2 = st.tabs([
        "Sans segmentation — baseline full-image",
        "Avec segmentation — crop→parent + feedback",
    ])

    with tab1:
        st.subheader("Résultats baseline full-image")

        if len(baseline_results) == 0:
            st.info("Aucun résultat baseline disponible.")
        else:
            cols_b = st.columns(5)

            for j, (p, s) in enumerate(baseline_results):
                c = cols_b[j % 5]

                if os.path.exists(p):
                    c.image(
                        Image.open(p).convert("RGB"),
                        caption=f"{j + 1}. score={s:.3f}\n{Path(p).name}",
                        use_container_width=True,
                    )
                else:
                    c.write(f"{j + 1}. {Path(p).name}")
                    c.caption("Fichier introuvable")

    with tab2:
        st.subheader("Top résultats segmentés crop→parent")

        if enable_feedback:
            st.caption(
                f"Itération feedback : {st.session_state.feedback_iteration} | "
                f"Prototypes positifs : {len(st.session_state.pos_dino_vecs)} | "
                f"Prototypes négatifs : {len(st.session_state.neg_dino_vecs)}"
            )

        if len(parents_ranked) == 0:
            st.info("Aucun parent retrouvé.")
            st.stop()

        selected_positive_rows = []
        selected_negative_rows = []

        st.caption(
            "Présentation en grille : les images affichées sont les images parentes, "
            "et la boîte rouge indique le crop responsable du meilleur score."
        )

        grid_cols = st.columns(5)

        for j, row in enumerate(parents_ranked):
            parent_path = row["parent_path"]
            parent_score = row["parent_score"]
            matches = row["matches"]

            c = grid_cols[j % 5]

            if not os.path.exists(parent_path):
                c.warning(f"Parent introuvable : {parent_path}")
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

            c.image(
                make_thumbnail(parent_disp, size=(320, 240)),
                caption=(
                    f"#{j + 1} | score={parent_score:.3f}\n"
                    f"{Path(parent_path).name}"
                ),
                use_container_width=True,
            )

            if enable_feedback:
                pos_key = (
                    f"seg_pos_{st.session_state.feedback_iteration}_"
                    f"{j}_{parent_path}"
                )

                neg_key = (
                    f"seg_neg_{st.session_state.feedback_iteration}_"
                    f"{j}_{parent_path}"
                )

                pos = c.checkbox("Pertinent", key=pos_key)
                neg = c.checkbox("Non pertinent", key=neg_key)

                if pos and neg:
                    c.warning("Choisis un seul statut.")
                elif pos:
                    selected_positive_rows.append(row)
                elif neg:
                    selected_negative_rows.append(row)

            with c.expander("Match local"):
                c.write(f"Parent : {parent_path}")

                for k_idx, (s, cp, bb) in enumerate(matches, start=1):
                    c.write(
                        f"match #{k_idx} | score={s:.3f} | crop={Path(cp).name}"
                    )

                    if os.path.exists(cp):
                        c.image(
                            make_thumbnail(Image.open(cp), size=(240, 180)),
                            caption=f"crop match #{k_idx}",
                            use_container_width=True,
                        )

        if enable_feedback:
            st.markdown("---")

            if st.button("Raffiner la recherche segmentée"):
                if len(selected_positive_rows) == 0 and len(selected_negative_rows) == 0:
                    st.warning("Sélectionne au moins un résultat pertinent ou non pertinent.")
                    st.stop()

                add_feedback_from_selected_rows(
                    selected_positive_rows=selected_positive_rows,
                    selected_negative_rows=selected_negative_rows,
                    crop_to_idx=crop_to_idx,
                    db_dino=db_dino,
                    db_clip=db_clip if mode in ("clip", "fusion") else None,
                )

                st.rerun()

            col_pos, col_neg = st.columns(2)

            with col_pos:
                st.subheader("Parents pertinents sélectionnés")

                if len(st.session_state.positive_parent_paths) == 0:
                    st.caption("Aucun pour l’instant.")
                else:
                    for p in st.session_state.positive_parent_paths:
                        st.caption(Path(p).name)

            with col_neg:
                st.subheader("Parents non pertinents sélectionnés")

                if len(st.session_state.negative_parent_paths) == 0:
                    st.caption("Aucun pour l’instant.")
                else:
                    for p in st.session_state.negative_parent_paths:
                        st.caption(Path(p).name)