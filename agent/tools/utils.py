import numpy as np
import os
import re
from functools import lru_cache
from pathlib import Path
from osgeo import gdal


def _resample_to_common_shape(band_a: np.ndarray, band_b: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    if band_a.shape == band_b.shape:
        return band_a, band_b
    from scipy.ndimage import zoom
    target_h = min(band_a.shape[0], band_b.shape[0])
    target_w = min(band_a.shape[1], band_b.shape[1])
    if band_a.shape != (target_h, target_w):
        band_a = zoom(band_a, (target_h / band_a.shape[0], target_w / band_a.shape[1]), order=1)
    if band_b.shape != (target_h, target_w):
        band_b = zoom(band_b, (target_h / band_b.shape[0], target_w / band_b.shape[1]), order=1)
    return band_a, band_b


def _try_generate_missing_index(target_path: Path) -> bool:
    """
    Best-effort generation of missing derived rasters (ndbi/ndvi) from downloaded band files.
    """
    try:
        name = target_path.name.lower()
        m_ndbi = re.match(r"^ndbi_(.+)\.tif$", name)
        m_ndvi = re.match(r"^ndvi_(.+)\.tif$", name)
        if not m_ndbi and not m_ndvi:
            return False

        period = (m_ndbi or m_ndvi).group(1)  # type: ignore[union-attr]
        directory = target_path.parent

        if m_ndbi:
            a_candidates = [directory / f"{period}_B11.tif", directory / f"{period}_B12.tif"]
            b_candidates = [directory / f"{period}_B8.tif", directory / f"{period}_B8A.tif"]
            formula = "ndbi"
        else:
            a_candidates = [directory / f"{period}_B8.tif", directory / f"{period}_B8A.tif"]
            b_candidates = [directory / f"{period}_B4.tif"]
            formula = "ndvi"

        band_a_path = next((p for p in a_candidates if p.is_file()), None)
        band_b_path = next((p for p in b_candidates if p.is_file()), None)
        if band_a_path is None or band_b_path is None:
            return False

        import rasterio

        with rasterio.open(band_a_path) as src_a:
            band_a = src_a.read(1).astype(np.float32)
            profile = src_a.profile.copy()
        with rasterio.open(band_b_path) as src_b:
            band_b = src_b.read(1).astype(np.float32)

        band_a, band_b = _resample_to_common_shape(band_a, band_b)
        denominator = band_a + band_b + 1e-6
        if formula == "ndbi":
            result = (band_a - band_b) / denominator
        else:
            result = (band_a - band_b) / denominator

        profile.update(dtype=rasterio.float32, nodata=-9999, compress="lzw", height=result.shape[0], width=result.shape[1])
        target_path.parent.mkdir(parents=True, exist_ok=True)
        with rasterio.open(target_path, "w", **profile) as dst:
            dst.write(result.astype(rasterio.float32), 1)
        return target_path.is_file()
    except Exception:
        return False


@lru_cache(maxsize=512)
def _resolve_existing_file_path(file_path: str) -> str:
    """
    Resolve file path robustly for tool calls.
    Supports:
    - direct absolute/relative paths
    - relative paths like "question3/ndbi_2023.tif" by searching workspace
    """
    raw_path = str(file_path or "").strip().strip('"').strip("'").strip("`")
    if not raw_path:
        raise RuntimeError("Failed to open empty file path")

    path_obj = Path(raw_path)
    if path_obj.is_file():
        return str(path_obj)
    if path_obj.is_absolute() and _try_generate_missing_index(path_obj):
        return str(path_obj)

    normalized = Path(os.path.normpath(raw_path))
    if normalized.is_file():
        return str(normalized)
    if normalized.is_absolute() and _try_generate_missing_index(normalized):
        return str(normalized)

    # Try workspace-relative resolution for relative paths.
    if not normalized.is_absolute():
        cwd = Path.cwd()

        direct = cwd / normalized
        if direct.is_file():
            return str(direct)
        if _try_generate_missing_index(direct):
            return str(direct)

        parts = [p for p in normalized.parts if p and p != "."]
        name = normalized.name
        matches: list[Path] = []
        max_candidates = 300
        suffix_variants: list[tuple[str, ...]] = []
        if parts:
            suffix_variants.append(tuple(parts))
            # Common LLM mismatch: "question2/..." vs actual "question_2/..."
            q_match = re.match(r"(?i)^question(\d+)$", parts[0])
            if q_match:
                normalized_question_dir = f"question_{q_match.group(1)}"
                suffix_variants.append(tuple([normalized_question_dir, *parts[1:]]))
        else:
            suffix_variants.append(tuple())

        search_roots: list[Path] = []
        eval_root = cwd / "evaluate_langchain"
        if eval_root.is_dir():
            search_roots.append(eval_root)
        benchmark_data_root = cwd / "qual" / "benchmark" / "data"
        if benchmark_data_root.is_dir():
            search_roots.append(benchmark_data_root)
        # Small fallback: current directory only (no recursive repo-wide scan).
        search_roots.append(cwd)

        if name:
            for root in search_roots:
                if len(matches) >= max_candidates:
                    break
                iterator = root.rglob(name) if root != cwd else cwd.glob(name)
                for candidate in iterator:
                    if not candidate.is_file():
                        continue
                    if parts:
                        matched_variant = False
                        for variant in suffix_variants:
                            if not variant:
                                continue
                            if len(candidate.parts) >= len(variant) and tuple(candidate.parts[-len(variant):]) == variant:
                                matched_variant = True
                                break
                        if not matched_variant:
                            continue
                        matches.append(candidate)
                        if len(matches) >= max_candidates:
                            break
                    elif not parts:
                        matches.append(candidate)
                        if len(matches) >= max_candidates:
                            break

        if matches:
            def _priority_key(p: Path):
                s = str(p).replace("/", "\\").lower()
                in_eval_out = ("evaluate_langchain\\" in s) and ("\\out\\" in s)
                return (1 if in_eval_out else 0, p.stat().st_mtime)

            matches.sort(key=_priority_key, reverse=True)
            return str(matches[0])

    raise RuntimeError(f"Failed to open {file_path}")


def read_image(file_path: str) -> np.ndarray:
    resolved_path = _resolve_existing_file_path(file_path)
    ds = gdal.Open(resolved_path)
    if ds is None:
        raise RuntimeError(f"Failed to open {file_path} (resolved: {resolved_path})")
    
    bands = ds.RasterCount
    if bands == 1:
        img = ds.GetRasterBand(1).ReadAsArray()
    else:
        img = np.stack([ds.GetRasterBand(i + 1).ReadAsArray() for i in range(bands)], axis=0)
        img = np.transpose(img, (1, 2, 0))

    ds = None
    return img


def read_image_uint8(file_path: str) -> np.ndarray:
    resolved_path = _resolve_existing_file_path(file_path)
    ds = gdal.Open(resolved_path)
    if ds is None:
        raise RuntimeError(f"Failed to open {file_path} (resolved: {resolved_path})")
    
    bands = ds.RasterCount
    if bands == 1:
        img = ds.GetRasterBand(1).ReadAsArray()
    else:
        img = np.stack([ds.GetRasterBand(i + 1).ReadAsArray() for i in range(bands)], axis=0)
        img = np.transpose(img, (1, 2, 0))

    ds = None

    img = img.astype(np.float32)
    min_val = np.min(img)
    max_val = np.max(img)

    if max_val > min_val:
        img = (img - min_val) / (max_val - min_val) * 255
    else:
        img = np.zeros_like(img)

    return img.astype(np.uint8)


def get_geotransform(file_path) -> tuple:
    resolved_path = _resolve_existing_file_path(file_path)
    ds = gdal.Open(resolved_path)
    if ds is None:
        raise RuntimeError(f"Failed to open {file_path} (resolved: {resolved_path})")
    geo = ds.GetGeoTransform()
    proj = ds.GetProjection()
    ds = None
    if geo == (0, 1.0, 0, 0, 0, 1.0):
        return None, None
    else:
        return geo, proj
