#!/usr/bin/env python3
# dupefinder.py
# Fixed-camera near-duplicate remover + diverse subset selector
# - Pure NumPy for cosine dedupe (no FAISS; avoids libomp clash on macOS)
# - Handles multiple extensions & recursion
# - Day/Night binning to preserve variety
# - Moves duplicates to a dupes directory; writes keep.txt

import os
import shutil
# Prevent OpenMP crash on macOS when multiple runtimes get loaded (Torch/OpenCV/etc.)
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("OMP_NUM_THREADS", "1")

import glob
from collections import defaultdict, deque

import numpy as np
from PIL import Image
from tqdm import tqdm
import imagehash

import torch
import open_clip

from skimage.metrics import structural_similarity as ssim
import cv2

# =======================
# Settings (tune these)
# =======================
# One or more roots to scan (recursive). You can add more folders here.
imgRoots = [
    "negatives",  # <-- change me
]

# File extensions to include (case-insensitive)
imgExts = [".jpg", ".jpeg", ".png", ".bmp"]

maxImages = None                 # None or int to cap during testing
phashHamming = 5                 # <=5 => near-duplicate by pHash
useSsim = True
ssimThreshold = 0.93             # higher = stricter (drop more)

# OpenCLIP model (use LAION weights to avoid QuickGELU warning)
clipModelName = "ViT-B-32"
clipPretrained = "laion2b_s34b_b79k"
embedBatchSize = 64
device = "cuda" if torch.cuda.is_available() else "cpu"

# Cosine similarity threshold for duplicate graph
dupCosine = 0.97                 # higher = stricter (drop more)
# Final diverse sample size; set None to keep everything post-dedupe
# Default to 700 as requested
finalKeepCount = 700            # e.g., 5000 or None

# Day/Night split
# Option A: Provide exemplar images; classification uses nearest prototype in (luma, saturation)
# Set absolute or relative paths here, or simply place files named
# 'ref_day.jpg' and 'ref_night.jpg' in the same folder as this script.
refDayImage = None      # e.g., "model_maker_2/ref_day.jpg"
refNightImage = None    # e.g., "model_maker_2/ref_night.jpg"
refWeights = (1.0, 1.0) # weights for (luma, saturation) distances

# Option B: Auto-detect thresholds using luminance and saturation.
autoNightSplit = True
# Night classification rule: if False, use (L <= thrL) AND (S <= thrS)
# AND is less aggressive and helps when IR makes luma bright but saturation low.
nightRuleOr = False
# Fallback thresholds if auto split is inconclusive
fallbackNightLumaMax = 60        # 0..255, higher if IR makes nights bright
fallbackNightSatMax = 30         # 0..255, IR nights often low saturation

# Output
keepFile = "keep.txt"
defaultDupesDir = "negative_dupes"

def rel_under_roots(path, roots):
    """Return a relative path under the dupes dir, namespaced by root basename.
    If no root matches, fall back to the file basename.
    """
    path = os.path.abspath(path)
    for r in roots:
        r_abs = os.path.abspath(r)
        try:
            rel = os.path.relpath(path, r_abs)
        except ValueError:
            continue
        if not rel.startswith(".."):
            return os.path.join(os.path.basename(os.path.normpath(r_abs)), rel)
    return os.path.basename(path)

# =======================
# Helpers
# =======================
def listImagesRecursive(roots, exts, limit=None):
    extsLower = {e.lower() for e in exts}
    files = []
    for root in roots:
        for path in glob.glob(os.path.join(root, "**", "*"), recursive=True):
            if not os.path.isfile(path):
                continue
            if os.path.getsize(path) <= 0:
                continue
            if os.path.splitext(path)[1].lower() in extsLower:
                files.append(os.path.abspath(path))
    files = sorted(set(files))
    if limit:
        files = files[:limit]
    return files

def imgToGrayLuma(path, maxSide=256):
    img = Image.open(path).convert("L")
    w, h = img.size
    scale = maxSide / max(w, h)
    if scale < 1.0:
        img = img.resize((int(w*scale), int(h*scale)), Image.BILINEAR)
    arr = np.array(img, dtype=np.uint8)
    return arr

def medianLuma(path):
    arr = imgToGrayLuma(path, maxSide=256)
    return float(np.median(arr))

def computePHashes(paths):
    hashes = []
    for p in tqdm(paths, desc="pHash"):
        try:
            h = imagehash.phash(Image.open(p).convert("RGB"))
        except Exception:
            h = None
        hashes.append(h)
    return hashes

def hamming(a, b):
    return None if (a is None or b is None) else (a - b)

def quickDedupeByPHash(paths, phashes, maxDist=5):
    keep = set()
    delete = set()
    visited = set()
    n = len(paths)

    for i in range(n):
        if i in visited:
            continue
        if phashes[i] is None:
            keep.add(i)
            visited.add(i)
            continue

        # Build a small cluster by Hamming distance
        q = deque([i])
        cluster = []
        while q:
            idx = q.popleft()
            if idx in visited:
                continue
            visited.add(idx)
            cluster.append(idx)
            # Brute force compare; for larger sets, index by hash prefixes
            for j in range(n):
                if j in visited:
                    continue
                d = hamming(phashes[idx], phashes[j])
                if d is not None and d <= maxDist:
                    q.append(j)

        if len(cluster) == 1 or not useSsim:
            rep = cluster[0]
            keep.add(rep)
            for k in cluster[1:]:
                delete.add(k)
        else:
            # Pick sharpest as representative; drop others only if SSIM is high
            sharpness = []
            for k in cluster:
                arr = imgToGrayLuma(paths[k], maxSide=256)
                sharpness.append(cv2.Laplacian(arr, cv2.CV_64F).var())
            rep = cluster[int(np.argmax(sharpness))]
            keep.add(rep)
            arrRep = imgToGrayLuma(paths[rep], maxSide=384)
            for k in cluster:
                if k == rep:
                    continue
                arrB = imgToGrayLuma(paths[k], maxSide=384)
                s = ssim(arrRep, arrB, data_range=255)
                if s >= ssimThreshold:
                    delete.add(k)
                else:
                    keep.add(k)

    keep = sorted(list(keep - delete))
    delete = sorted(list(delete))
    return keep, delete

def loadOpenClip(modelName="ViT-B-32", pretrained="laion2b_s34b_b79k", device="cpu"):
    model, _, preprocess = open_clip.create_model_and_transforms(modelName, pretrained=pretrained, device=device)
    model.eval()
    return model, preprocess

@torch.no_grad()
def computeEmbeddings(paths, model, preprocess, device="cpu", batchSize=64):
    embs = []
    for i in tqdm(range(0, len(paths), batchSize), desc="Embeddings"):
        batch = []
        for p in paths[i:i+batchSize]:
            try:
                img = Image.open(p).convert("RGB")
                batch.append(preprocess(img))
            except Exception:
                batch.append(torch.zeros(3, 224, 224))
        x = torch.stack(batch).to(device)
        feats = model.encode_image(x)
        feats = feats / feats.norm(dim=-1, keepdim=True)
        embs.append(feats.cpu().numpy().astype(np.float32))
    return np.concatenate(embs, axis=0)

def cosineDedup(paths, embs, simThresh=0.97):
    # Pure NumPy, no FAISS. embs are L2-normalized.
    sims = embs @ embs.T  # (n, n)
    n = sims.shape[0]
    visited = [False]*n
    keep, delete = [], set()

    for i in range(n):
        if visited[i]:
            continue
        neighbors = [i]
        visited[i] = True
        # collect neighbors above threshold (one-hop)
        for j in range(n):
            if j == i or visited[j]:
                continue
            if sims[i, j] >= simThresh:
                neighbors.append(j)
                visited[j] = True
        if len(neighbors) == 1:
            keep.append(i)
        else:
            sharpness = []
            for k in neighbors:
                arr = imgToGrayLuma(paths[k], maxSide=256)
                sharpness.append(cv2.Laplacian(arr, cv2.CV_64F).var())
            rep = neighbors[int(np.argmax(sharpness))]
            keep.append(rep)
            for k in neighbors:
                if k != rep:
                    delete.add(k)
    return sorted(keep), sorted(list(delete))

def kCenterGreedy(embs, targetK, initIdx=None):
    n = embs.shape[0]
    if targetK is None or targetK >= n:
        return list(range(n))
    # Seed with provided indices or a random one
    if initIdx is None:
        chosen = [int(np.random.randint(0, n))]
    else:
        if isinstance(initIdx, (list, tuple, np.ndarray)):
            chosen = [int(i) for i in initIdx if 0 <= int(i) < n]
            if not chosen:
                chosen = [int(np.random.randint(0, n))]
        else:
            chosen = [int(initIdx)]
    # distances to nearest chosen center (cosine distance)
    d = np.full(n, np.inf, dtype=np.float32)
    for c in chosen:
        sim = embs @ embs[c]
        dist = np.clip(1.0 - sim, 0.0, 2.0)
        d = np.minimum(d, dist)
    # Greedily add until targetK
    for _ in range(len(chosen), min(targetK, n)):
        nextIdx = int(np.argmax(d))
        chosen.append(nextIdx)
        sim = embs @ embs[nextIdx]
        dist = np.clip(1.0 - sim, 0.0, 2.0)
        d = np.minimum(d, dist)
    return chosen[:targetK]

def medianSaturation(path, maxSide=256):
    try:
        img = Image.open(path).convert("RGB")
        w, h = img.size
        scale = maxSide / max(w, h)
        if scale < 1.0:
            img = img.resize((int(w*scale), int(h*scale)), Image.BILINEAR)
        arr = np.array(img, dtype=np.uint8)
        hsv = cv2.cvtColor(arr, cv2.COLOR_RGB2HSV)
        return float(np.median(hsv[..., 1]))
    except Exception:
        return 128.0

def _auto_threshold(vals, bins=64):
    vals = np.asarray(vals, dtype=np.float32)
    if vals.size == 0:
        return None
    hist, edges = np.histogram(vals, bins=bins, range=(0, 255))
    mid = bins // 2
    if hist.sum() == 0:
        return None
    p1 = int(np.argmax(hist[:mid]))
    p2 = mid + int(np.argmax(hist[mid:]))
    lo, hi = (p1, p2) if p1 < p2 else (p2, p1)
    if hi - lo <= 2:
        return None
    valley_idx = int(np.argmin(hist[lo:hi])) + lo
    thr = 0.5 * (edges[valley_idx] + edges[valley_idx+1])
    return float(thr)

def splitDayNight(paths):
    # Resolve reference images from explicit config or default filenames
    script_dir = os.path.dirname(os.path.abspath(__file__))
    default_day = os.path.join(script_dir, "ref_day.jpg")
    default_night = os.path.join(script_dir, "ref_night.jpg")
    day_ref_path = refDayImage if refDayImage and os.path.isfile(refDayImage) else (default_day if os.path.isfile(default_day) else None)
    night_ref_path = refNightImage if refNightImage and os.path.isfile(refNightImage) else (default_night if os.path.isfile(default_night) else None)

    # Reference-based classification (if both refs available)
    if day_ref_path and night_ref_path:
        try:
            Ld = medianLuma(day_ref_path); Sd = medianSaturation(day_ref_path)
            Ln = medianLuma(night_ref_path); Sn = medianSaturation(night_ref_path)
            wL, wS = refWeights
            print(f"Using reference-based day/night split:\n  day  ref: '{day_ref_path}'  L={Ld:.1f} S={Sd:.1f}\n  night ref: '{night_ref_path}' L={Ln:.1f} S={Sn:.1f}")
            dayIdx, nightIdx = [], []
            for i, p in enumerate(paths):
                L = medianLuma(p); S = medianSaturation(p)
                dd = wL*(L - Ld)*(L - Ld) + wS*(S - Sd)*(S - Sd)
                dn = wL*(L - Ln)*(L - Ln) + wS*(S - Sn)*(S - Sn)
                if dn < dd:
                    nightIdx.append(i)
                else:
                    dayIdx.append(i)
            print(f"Ref-based split: day={len(dayIdx)} night={len(nightIdx)}")
            return dayIdx, nightIdx
        except Exception as e:
            print(f"Ref-based day/night split failed; falling back to auto. Error: {e}")

    if not autoNightSplit:
        dayIdx, nightIdx = [], []
        for i, p in enumerate(paths):
            med = medianLuma(p)
            if med <= fallbackNightLumaMax:
                nightIdx.append(i)
            else:
                dayIdx.append(i)
        return dayIdx, nightIdx

    # Auto mode: use luminance and saturation with valley thresholds
    lumas, sats = [], []
    for p in paths:
        lumas.append(medianLuma(p))
        sats.append(medianSaturation(p))
    thrL = _auto_threshold(lumas)
    thrS = _auto_threshold(sats)
    if thrL is None:
        thrL = float(fallbackNightLumaMax)
    if thrS is None:
        thrS = float(fallbackNightSatMax)

    dayIdx, nightIdx = [], []
    for i, (L, S) in enumerate(zip(lumas, sats)):
        isNight = (L <= thrL) or (S <= thrS) if nightRuleOr else ((L <= thrL) and (S <= thrS))
        if isNight:
            nightIdx.append(i)
        else:
            dayIdx.append(i)
    return dayIdx, nightIdx

# =======================
# Main
# =======================
def main():
    dupesDir = defaultDupesDir

    paths = listImagesRecursive(imgRoots, imgExts, maxImages)
    print(f"Found {len(paths)} images")
    if len(paths) == 0:
        # Still write empty files for consistency
        open(keepFile, "w").close()
        print("No images found. Wrote empty keep.txt")
        return

    phashes = computePHashes(paths)
    keepIdx1, delIdx1 = quickDedupeByPHash(paths, phashes, maxDist=phashHamming)
    paths1 = [paths[i] for i in keepIdx1]
    print(f"After pHash/SSIM: keep {len(paths1)}, delete {len(delIdx1)}")

    if len(paths1) == 0:
        # Everything deduped by pHash/SSIM
        with open(keepFile, "w") as f:
            pass
        # Move all deleted images into dupesDir
        moved = 0
        for i in delIdx1:
            src = paths[i]
            rel = rel_under_roots(src, imgRoots)
            dst = os.path.join(dupesDir, rel)
            os.makedirs(os.path.dirname(dst), exist_ok=True)
            try:
                shutil.move(src, dst)
                moved += 1
            except Exception as e:
                print(f"Failed to move {src} -> {dst}: {e}")
        print(f"Final keep: 0; moved {moved} images to '{dupesDir}'")
        print(f"Wrote {keepFile}")
        return

    # Split day/night to preserve variety
    dayRel, nightRel = splitDayNight(paths1)

    # Load CLIP
    model, preprocess = loadOpenClip(clipModelName, clipPretrained, device)

    keptAllRel = []
    deletedAllRel = set()

    for label, relIdx in [("day", dayRel), ("night", nightRel)]:
        if not relIdx:
            print(f"[{label}] no images in this bin.")
            continue
        subPaths = [paths1[i] for i in relIdx]
        embs = computeEmbeddings(subPaths, model, preprocess, device, embedBatchSize)

        keepRel, delRel = cosineDedup(subPaths, embs, simThresh=dupCosine)
        print(f"[{label}] cosine-dedupe: keep {len(keepRel)}, delete {len(delRel)}")

        # Optional: diverse subset inside bin
        if finalKeepCount is not None:
            # Allocate capacity proportional to bin size after dedupe
            totalAfter = len(keepRel)
            if totalAfter > 0:
                # proportion = totalAfter / sum over bins; we do it lazily at the end
                pass

        # Accumulate kept/deleted (relative to paths1)
        keptAllRel += [relIdx[i] for i in keepRel]
        for j in delRel:
            deletedAllRel.add(relIdx[j])

    keptAllRel = sorted(set(keptAllRel))
    deletedAllRel = sorted(set(deletedAllRel))

    # If finalKeepCount is requested, adjust kept set to match target size
    if finalKeepCount is not None:
        target = min(finalKeepCount, len(paths))
        if len(keptAllRel) > target:
            # Downsample kept set to target using k-center on kept only
            keptPaths = [paths1[i] for i in keptAllRel]
            embsKept = computeEmbeddings(keptPaths, model, preprocess, device, embedBatchSize)
            sel = kCenterGreedy(embsKept, target)
            keptAllRel = [keptAllRel[s] for s in sel]
        elif len(keptAllRel) < target:
            # Top up by selecting additional diverse images from all originals
            # using k-center seeded with current kept absolute indices
            # Map current kept to absolute indices before topping up
            keepAbs_seed = sorted(keepIdx1[i] for i in keptAllRel)
            # Compute embeddings for all originals
            embsAll = computeEmbeddings(paths, model, preprocess, device, embedBatchSize)
            selAbs = kCenterGreedy(embsAll, targetK=target, initIdx=keepAbs_seed)
            # Convert absolute selection back into absolute indices directly
            keepAbs_final = sorted(selAbs)
            # Overwrite pipeline to finalize using these absolute indices
            keepAbs = keepAbs_final
            deleteAbs = set(range(len(paths))) - set(keepAbs)
            # Short-circuit the usual mapping below by jumping to output section
            # Output keep file and move deletes
            with open(keepFile, "w") as f:
                for i in keepAbs:
                    f.write(paths[i] + "\n")
            moved = 0
            for i in sorted(deleteAbs):
                src = paths[i]
                rel = rel_under_roots(src, imgRoots)
                dst = os.path.join(dupesDir, rel)
                os.makedirs(os.path.dirname(dst), exist_ok=True)
                try:
                    shutil.move(src, dst)
                    moved += 1
                except Exception as e:
                    print(f"Failed to move {src} -> {dst}: {e}")
            print(f"Final keep: {len(keepAbs)}; moved {moved} images to '{dupesDir}'")
            print(f"Wrote {keepFile}")
            return

    # -------------------------------
    # Map RELATIVE indices back to ORIGINAL 'paths' and write files
    # -------------------------------
    keepAbs = sorted(keepIdx1[i] for i in keptAllRel)
    # Add everything that was dropped by pHash/SSIM and by cosine to deletes
    deleteAbs = set(range(len(paths))) - set(keepAbs)
    # Sanity: every original index is either kept or deleted
    assert len(keepAbs) + len(deleteAbs) == len(paths)

    with open(keepFile, "w") as f:
        for i in keepAbs:
            f.write(paths[i] + "\n")

    # Move deleted images into dupesDir
    moved = 0
    for i in sorted(deleteAbs):
        src = paths[i]
        rel = rel_under_roots(src, imgRoots)
        dst = os.path.join(dupesDir, rel)
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        try:
            shutil.move(src, dst)
            moved += 1
        except Exception as e:
            print(f"Failed to move {src} -> {dst}: {e}")

    print(f"Final keep: {len(keepAbs)}; moved {moved} images to '{dupesDir}'")
    print(f"Wrote {keepFile}")

if __name__ == "__main__":
    main()
