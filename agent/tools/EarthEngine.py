"""
Earth Engine MCP Server - Google Earth Engine REST API integration for Earth-Agent

This MCP server provides tools to search and download satellite imagery from
Google Earth Engine collections using REST API.

Tools:
- search_images: Search for satellite images by location, date, and cloud coverage
- download_band: Download a single band from an image as GeoTIFF
- download_bands: Batch download multiple bands from an image
- get_image_metadata: Get detailed information about an image
- list_collections: List available Earth Engine collections

Requires:
- Google Cloud service account with Earth Engine access
- Service account key JSON file
- Environment variables: GEE_SERVICE_ACCOUNT_KEY or GEE_SERVICE_ACCOUNT_KEY_JSON
"""

import os
import sys
import json
import io
import argparse
import hashlib
import builtins
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
import urllib.parse

# IMPORTANT: MCP stdio protocol expects ONLY JSON-RPC on stdout.
# Route all diagnostic prints to stderr to avoid breaking the client parser.
def print(*args, **kwargs):  # type: ignore[override]
    kwargs.setdefault("file", sys.stderr)
    kwargs.setdefault("flush", True)
    return builtins.print(*args, **kwargs)

# Third-party imports
import numpy as np
import rasterio
from rasterio.transform import from_bounds
from rasterio.crs import CRS

# Google Auth
from google.auth.transport.requests import AuthorizedSession
from google.oauth2 import service_account

# Redis for caching
try:
    import redis
    REDIS_AVAILABLE = True
except ImportError:
    REDIS_AVAILABLE = False
    print("Warning: redis package not installed. Caching will be disabled.")
    print("Install with: pip install redis")

from dev_tag import dev_mcp_tool

# FastMCP
from fastmcp import FastMCP

# Parse command line arguments
parser = argparse.ArgumentParser(description="Earth Engine MCP Server")
parser.add_argument(
    "--temp_dir", type=str, required=True, help="Temporary directory for output files"
)
parser.add_argument(
    "--gee_key", type=str, default=None, help="Path to GEE service account key (optional, overrides env)"
)
args = parser.parse_args()

TEMP_DIR = Path(args.temp_dir)
TEMP_DIR.mkdir(parents=True, exist_ok=True)

# If GEE key passed via args, set it in env for _get_session to use
if args.gee_key:
    os.environ['GEE_SERVICE_ACCOUNT_KEY'] = args.gee_key
    print(f"Using GEE key from args: {args.gee_key}")

# Initialize FastMCP
mcp = FastMCP("EarthEngine")

# Global session (will be initialized on first use)
_session: Optional[AuthorizedSession] = None

# Redis client (will be initialized on first use)
_redis_client: Optional[Any] = None


def _create_png_preview(tif_path: Path, png_path: Path, band_name: str = None) -> Tuple[bool, str]:
    """
    Create PNG preview from GeoTIFF file.
    
    Parameters:
        tif_path: Path to input GeoTIFF file
        png_path: Path to output PNG file
        band_name: Optional band name for title
    Returns:
        tuple[bool, str]: (created, backend/reason)
    """
    # Read the GeoTIFF once; both backends use the same source data.
    with rasterio.open(tif_path) as src:
        data = src.read(1)
        nodata = src.nodata

    # Preferred backend: matplotlib (if available).
    try:
        import matplotlib
        matplotlib.use('Agg')  # Non-interactive backend
        import matplotlib.pyplot as plt

        # Mask nodata values
        if nodata is not None:
            data_plot = np.ma.masked_equal(data, nodata)
        else:
            data_plot = data

        fig, ax = plt.subplots(figsize=(10, 10))
        im = ax.imshow(data_plot, cmap='viridis')
        plt.colorbar(im, ax=ax, label='Value')
        title = f"Band: {band_name}" if band_name else "Preview"
        title += f"\nShape: {data_plot.shape}, Min: {data_plot.min():.4f}, Max: {data_plot.max():.4f}"
        ax.set_title(title)
        plt.savefig(png_path, dpi=150, bbox_inches='tight')
        plt.close(fig)
        return True, "matplotlib"
    except ImportError:
        # Fallback backend: Pillow (grayscale quicklook).
        try:
            from PIL import Image

            arr = data.astype(np.float32)

            if nodata is not None:
                arr[arr == nodata] = np.nan

            valid_mask = np.isfinite(arr)
            if np.any(valid_mask):
                valid_values = arr[valid_mask]
                vmin = np.percentile(valid_values, 2)
                vmax = np.percentile(valid_values, 98)
                if not np.isfinite(vmin):
                    vmin = float(np.nanmin(valid_values))
                if not np.isfinite(vmax):
                    vmax = float(np.nanmax(valid_values))
                if vmax <= vmin:
                    vmax = vmin + 1.0
                norm = np.clip((arr - vmin) / (vmax - vmin), 0.0, 1.0)
                norm = np.where(valid_mask, norm, 0.0)
            else:
                norm = np.zeros_like(arr, dtype=np.float32)

            preview = (norm * 255).astype(np.uint8)
            Image.fromarray(preview, mode='L').save(png_path)
            return True, "pillow"
        except Exception as e:
            return False, f"preview backend unavailable ({e})"
    except Exception as e:
        return False, f"matplotlib preview failed ({e})"
    

def _get_redis_client() -> Optional[Any]:
    """Get or create Redis client for caching."""
    global _redis_client
    
    if not REDIS_AVAILABLE:
        return None
    
    if _redis_client is not None:
        return _redis_client
    
    redis_url = os.getenv("REDIS_URL", "redis://localhost:6379")
    
    try:
        _redis_client = redis.from_url(redis_url, decode_responses=False)
        # Test connection
        _redis_client.ping()
        print(f"[INFO] Redis cache enabled: {redis_url}")
        return _redis_client
    except Exception as e:
        print(f"Warning: Redis connection failed: {e}")
        print("Caching will be disabled for this session.")
        return None


def _cache_key(prefix: str, *args, **kwargs) -> str:
    """Generate cache key from function arguments."""
    # Create deterministic string from args and kwargs
    key_data = f"{prefix}:" + json.dumps({"args": args, "kwargs": kwargs}, sort_keys=True)
    # Hash to keep key length reasonable
    return f"earthengine:{hashlib.sha256(key_data.encode()).hexdigest()}"


def _get_cached(cache_key: str) -> Optional[bytes]:
    """Get data from cache."""
    redis_client = _get_redis_client()
    if redis_client is None:
        return None
    
    try:
        data = redis_client.get(cache_key)
        if data:
            print(f"[INFO] Cache hit: {cache_key[:50]}...")
        return data
    except Exception as e:
        print(f"Warning: Cache read error: {e}")
        return None


def _set_cached(cache_key: str, data: bytes, ttl: int = 86400) -> None:
    """Set data in cache with TTL (default 24 hours)."""
    redis_client = _get_redis_client()
    if redis_client is None:
        return
    
    try:
        redis_client.setex(cache_key, ttl, data)
        print(f"[INFO] Cached: {cache_key[:50]}...")
    except Exception as e:
        print(f"Warning: Cache write error: {e}")


def _get_session() -> AuthorizedSession:
    """Get or create authenticated Earth Engine session."""
    global _session

    if _session is not None:
        return _session

    # Try to get credentials from environment
    key_path = os.getenv("GEE_SERVICE_ACCOUNT_KEY")
    key_json_str = os.getenv("GEE_SERVICE_ACCOUNT_KEY_JSON")

    if key_path and os.path.exists(key_path):
        # Load from file
        credentials = service_account.Credentials.from_service_account_file(key_path)
    elif key_json_str:
        # Load from JSON string
        key_data = json.loads(key_json_str)
        credentials = service_account.Credentials.from_service_account_info(key_data)
    else:
        raise ValueError(
            "Earth Engine credentials not found. Please set either:\n"
            "  GEE_SERVICE_ACCOUNT_KEY=/path/to/key.json\n"
            "or\n"
            '  GEE_SERVICE_ACCOUNT_KEY_JSON=\'{"type":"service_account",...}\''
        )

    # Create scoped credentials
    scoped_credentials = credentials.with_scopes(
        ["https://www.googleapis.com/auth/cloud-platform"]
    )

    _session = AuthorizedSession(scoped_credentials)
    return _session


# Popular Earth Engine collections with metadata and use case recommendations
COLLECTIONS = {
    "sentinel2": {
        "id": "COPERNICUS/S2_SR_HARMONIZED",
        "name": "Sentinel-2 MSI Surface Reflectance",
        "resolution": 10,
        "revisit_days": 5,
        "bands": ["B2", "B3", "B4", "B5", "B6", "B7", "B8", "B8A", "B11", "B12"],
        "cloud_property": "CLOUDY_PIXEL_PERCENTAGE",
        "best_for": [
            "Detailed urban analysis (buildings, roads, small features)",
            "High-precision vegetation indices (NDVI, EVI)",
            "Built-up territory proxy from SWIR + NIR",
            "Visible-band optical water-quality proxies",
            "Small-scale land cover mapping",
            "Monitoring fine-scale changes"
        ],
        "use_when": "Use when the task asks for Sentinel-2, high-detail city/vegetation/water analysis, or comparable optical scenes with SWIR/NIR/Red/Green bands",
        "advantages": "Best spatial resolution, excellent for urban and agricultural monitoring",
        "limitations": "Lower temporal frequency than HLS, larger file sizes. No thermal LST band: B10 is cirrus, B11/B12 are SWIR.",
        "selection_priority": "Prefer this collection for benchmark optical questions unless the question explicitly asks Landsat/HLS/MODIS or thermal LST.",
        "common_indices": {
            "NDVI": {"bands": ["B8", "B4"], "description": "Vegetation health (NIR - Red)"},
            "NDBI": {"bands": ["B11", "B8"], "description": "Built-up area (SWIR - NIR)"},
            "NDWI": {"bands": ["B3", "B8"], "description": "Water bodies (Green - NIR)"},
            "visible_turbidity_proxy": {"bands": ["B4"], "description": "Optical turbidity proxy from visible reflectance; use turbidity tool, not NDWI area share"}
        }
    },
    "landsat8": {
        "id": "LANDSAT/LC08/C02/T1_L2",
        "name": "Landsat 8 Collection 2 Tier 1 Level 2",
        "resolution": 30,
        "revisit_days": 16,
        "bands": ["SR_B1", "SR_B2", "SR_B3", "SR_B4", "SR_B5", "SR_B6", "SR_B7", "ST_B10", "QA_PIXEL"],
        "cloud_property": "CLOUD_COVER",
        "best_for": [
            "Long-term monitoring (consistent with Landsat archive since 1972)",
            "Historical change analysis",
            "Moderate-resolution land cover classification",
            "Land Surface Temperature (LST) and urban heat-island analysis from ST_B10"
        ],
        "use_when": "Use for thermal/LST questions, heat-island comparisons, or historical Landsat-only analyses",
        "advantages": "Long historical archive, well-documented, consistent calibration",
        "limitations": "16-day revisit time means gaps in temporal coverage; 30m optical resolution is coarser than Sentinel-2",
        "common_indices": {
            "NDVI": {"bands": ["SR_B5", "SR_B4"], "description": "Vegetation health (NIR - Red)"},
            "NDBI": {"bands": ["SR_B6", "SR_B5"], "description": "Built-up area (SWIR1 - NIR)"},
            "NDWI": {"bands": ["SR_B3", "SR_B5"], "description": "Water bodies (Green - NIR)"},
            "LST": {"bands": ["ST_B10"], "description": "USGS Level-2 surface temperature band for LST/heat-island tasks"}
        }
    },
    "landsat9": {
        "id": "LANDSAT/LC09/C02/T1_L2",
        "name": "Landsat 9 Collection 2 Tier 1 Level 2",
        "resolution": 30,
        "revisit_days": 16,
        "bands": ["SR_B1", "SR_B2", "SR_B3", "SR_B4", "SR_B5", "SR_B6", "SR_B7", "ST_B10", "QA_PIXEL"],
        "cloud_property": "CLOUD_COVER",
        "best_for": [
            "Recent monitoring (since 2021)",
            "Continuation of Landsat 8 time series",
            "Improved radiometric quality",
            "Recent Land Surface Temperature (LST) and heat-island analysis from ST_B10"
        ],
        "use_when": "Use for recent thermal/LST questions or continuation of Landsat 8 thermal/optical series",
        "advantages": "Improved radiometry over Landsat 8, identical band specifications",
        "limitations": "Shorter archive (only since 2021), 16-day revisit",
        "common_indices": {
            "NDVI": {"bands": ["SR_B5", "SR_B4"], "description": "Vegetation health (NIR - Red)"},
            "NDBI": {"bands": ["SR_B6", "SR_B5"], "description": "Built-up area (SWIR1 - NIR)"},
            "NDWI": {"bands": ["SR_B3", "SR_B5"], "description": "Water bodies (Green - NIR)"},
            "LST": {"bands": ["ST_B10"], "description": "USGS Level-2 surface temperature band for LST/heat-island tasks"}
        }
    },
    "hls_landsat": {
        "id": "NASA/HLS/HLSL30/v002",
        "name": "HLS Landsat 8/9 Surface Reflectance",
        "resolution": 30,
        "revisit_days": "2-3",
        "bands": ["B1", "B2", "B3", "B4", "B5", "B6", "B7", "B9", "B10", "B11"],
        "cloud_property": "CLOUD_COVERAGE",
        "best_for": [
            "Time series analysis and change detection",
            "Seasonal vegetation monitoring",
            "Multi-temporal analysis (frequent observations)",
            "Harmonized multi-sensor workflows"
        ],
        "use_when": "Use when harmonized 30m time series is more important than native Sentinel-2 detail, or when the question explicitly asks HLS",
        "advantages": "Best temporal coverage (combined L8+L9, harmonized with Sentinel-2), ready for time series",
        "limitations": "30m resolution (not as detailed as Sentinel-2). Do not switch to HLS for a benchmark question that explicitly asks Sentinel-2 unless Sentinel-2 search fails after date/cloud retries.",
        "recommended": False,
        "common_indices": {
            "NDVI": {"bands": ["B5", "B4"], "description": "Vegetation health (NIR - Red)"},
            "NDBI": {"bands": ["B6", "B5"], "description": "Built-up area (SWIR1 - NIR)"},
            "NDWI": {"bands": ["B3", "B5"], "description": "Water bodies (Green - NIR)"}
        }
    },
    "hls_sentinel": {
        "id": "NASA/HLS/HLSS30/v002",
        "name": "HLS Sentinel-2 Surface Reflectance",
        "resolution": 30,
        "revisit_days": "2-3",
        "bands": [
            "B1",
            "B2",
            "B3",
            "B4",
            "B5",
            "B6",
            "B7",
            "B8",
            "B8A",
            "B9",
            "B10",
            "B11",
            "B12",
        ],
        "cloud_property": "CLOUD_COVERAGE",
        "best_for": [
            "Time series with Sentinel-2 data quality",
            "Harmonized workflows combining Landsat and Sentinel",
            "Frequent monitoring at 30m resolution"
        ],
        "use_when": "Use when harmonized 30m Sentinel-like time series is required, not when native 10m Sentinel-2 Surface Reflectance is requested",
        "advantages": "Sentinel-2 bands at harmonized 30m, excellent temporal coverage",
        "limitations": "Resampled from native 10m (some detail loss). No thermal LST band; B10 is cirrus.",
        "common_indices": {
            "NDVI": {"bands": ["B8", "B4"], "description": "Vegetation health (NIR - Red)"},
            "NDBI": {"bands": ["B11", "B8"], "description": "Built-up area (SWIR1 - NIR)"},
            "NDWI": {"bands": ["B3", "B8"], "description": "Water bodies (Green - NIR)"}
        }
    },
    "modis": {
        "id": "MODIS/006/MOD09A1",
        "name": "MODIS Surface Reflectance 8-Day Global 500m",
        "resolution": 500,
        "revisit_days": "Daily (8-day composite)",
        "bands": [
            "sur_refl_b01",
            "sur_refl_b02",
            "sur_refl_b03",
            "sur_refl_b04",
            "sur_refl_b05",
            "sur_refl_b06",
            "sur_refl_b07",
        ],
        "cloud_property": None,
        "best_for": [
            "Continental/global scale monitoring",
            "Large-area rapid assessment",
            "Daily monitoring where spatial detail is not critical"
        ],
        "use_when": "You are working at regional/continental scale and need daily updates",
        "advantages": "Daily coverage, long archive, global coverage",
        "limitations": "Very coarse resolution (500m) - NOT suitable for detailed analysis",
        "not_recommended_for": [
            "Urban analysis",
            "Small features detection",
            "Detailed vegetation mapping",
            "Anything requiring < 100m resolution"
        ],
        "common_indices": {
            "NDVI": {"bands": ["sur_refl_b02", "sur_refl_b01"], "description": "Vegetation health (NIR - Red)"}
        }
    },
}


@dev_mcp_tool(
    mcp,
    description="""
List available Earth Engine satellite collections with detailed metadata and use case recommendations.

**IMPORTANT: Call this tool FIRST** before using search_images or download_band to understand which 
collection is best for your specific analysis task.

These recommendations are collection-selection guidance for the data acquisition
agent. They must not be treated as a hidden answer key: if the user/question gives
date, cloud, or season constraints, preserve those constraints and choose the
physically suitable collection inside them.

Selection guidance:
    - If the user specifies a collection family (Sentinel-2, Landsat, MODIS, HLS),
      keep that collection family for all compared periods. Do not switch from
      native Sentinel-2 to HLS just because HLS has more scenes, unless the
      requested collection has no usable images after date/cloud retries.
    - For multi-year comparisons, use the same collection, comparable season,
      and comparable cloud threshold across all periods. If the question gives
      target dates, search narrow windows around those dates first.
    - Built-up area/share: choose optical collections with SWIR + NIR bands
      (e.g. Sentinel-2 B11/B12 + B8, Landsat/HLS SWIR + NIR).
    - Vegetation area/share: choose optical collections with NIR + Red.
    - Water area/share: choose optical collections with Green + NIR/SWIR according
      to the chosen water index.
    - Water turbidity / optical turbidity proxy: choose visible reflectance data
      for turbidity tools. Do not replace turbidity with water-area NDWI unless
      the question asks for water area.
    - Surface temperature / heat island: choose Landsat Collection 2 L2 ST_B10
      or another true thermal/LST band.
      Sentinel-2 B10 is cirrus and Sentinel-2 B11/B12 are SWIR, not thermal.

Returns:
    dict: Detailed information about each collection including:
        - id: Full Earth Engine collection ID
        - name: Human-readable collection name
        - resolution: Spatial resolution in meters
        - revisit_days: How often the satellite revisits the same location
        - bands: List of available spectral bands
        - best_for: List of recommended use cases
        - use_when: Guidance on when to choose this collection
        - advantages: Key benefits of this collection
        - limitations: Important constraints to be aware of
        - common_indices: Spectral indices (NDVI, NDBI, NDWI) with band mappings
        - recommended: (optional) True if this is a recommended choice for general tasks
        - not_recommended_for: (optional) Tasks where this collection should NOT be used

**Workflow:**
1. Call list_collections() to see all available options with their characteristics
2. Read the metadata to choose the most appropriate collection for your task
3. Use search_images(collection_alias, bbox, dates) to find available imagery
4. Use download_band/download_bands to get the data

Example:
    collections = list_collections()
    # Returns: {"sentinel2": {...}, "hls_landsat": {...}, ...}
    # Each collection has detailed metadata to help you decide
""",
)
def list_collections() -> dict:
    """List available Earth Engine collections with detailed use case recommendations."""
    return COLLECTIONS


@dev_mcp_tool(
    mcp,
    description="""
Search for satellite images in Google Earth Engine collection.

This tool only searches metadata; it does not download pixels. Use it after
choosing a collection with list_collections. Prefer low cloud coverage, but if
the result is empty retry with a wider date window, higher cloud_cover_max, or a
different physically suitable collection.

When a target date is provided, the best scene is not always the globally lowest
cloud scene in the year. First search a narrow window around the target date,
then expand the window while keeping comparable season and collection.

Comparison guidance:
    - Keep the same collection across all compared years/periods.
    - If target dates are specified, search around those dates first rather than
      selecting a different season just because cloud coverage is lower.
    - Start with cloud_cover_max=20 when possible. If no scenes are found, retry
      50, then 80, before changing collection family.
    - For benchmark questions that say Sentinel-2 Surface Reflectance, use
      collection="sentinel2" first and keep it unless no suitable scenes exist.

Parameters:
    collection (str): Collection alias (e.g., "sentinel2", "landsat8", "hls_landsat") or full ID
    bbox (list[float]): Bounding box [min_lon, min_lat, max_lon, max_lat]
    start_date (str): Start date in ISO format "YYYY-MM-DD" or "YYYY-MM-DDTHH:MM:SSZ"
    end_date (str): End date in ISO format "YYYY-MM-DD" or "YYYY-MM-DDTHH:MM:SSZ"
    cloud_cover_max (float): Maximum cloud coverage percentage (default: 20)
    limit (int): Maximum number of images to return (default: 10)

Returns:
    list[dict]: List of images with metadata:
        - id: Full image ID
        - date: Acquisition date
        - cloud_coverage: Cloud coverage percentage
        - bands: List of available band names
        
Example:
    search_images("sentinel2", [37.6, 55.7, 37.7, 55.8], "2023-06-01", "2023-08-31", 10)
""",
)
def search_images(
    collection: str,
    bbox: List[float],
    start_date: str,
    end_date: str,
    cloud_cover_max: float = 20.0,
    limit: int = 10,
) -> List[Dict[str, Any]]:
    """Search for satellite images in Earth Engine."""

    session = _get_session()

    # Get collection ID
    if collection in COLLECTIONS:
        collection_id = COLLECTIONS[collection]["id"]
        cloud_property = COLLECTIONS[collection].get("cloud_property")
    else:
        collection_id = collection
        cloud_property = "CLOUD_COVERAGE"  # default

    # Build AOI polygon
    min_lon, min_lat, max_lon, max_lat = bbox
    aoi = {
        "type": "Polygon",
        "coordinates": [
            [
                [min_lon, min_lat],
                [min_lon, max_lat],
                [max_lon, max_lat],
                [max_lon, min_lat],
                [min_lon, min_lat],
            ]
        ],
    }

    # Ensure dates have time component
    if "T" not in start_date:
        start_date += "T00:00:00.000Z"
    if "T" not in end_date:
        end_date += "T23:59:59.999Z"

    # Build request
    project = "projects/earthengine-public"
    collection_name = f"{project}/assets/{collection_id}"

    params = {
        "startTime": start_date,
        "endTime": end_date,
        "region": json.dumps(aoi),
    }

    if cloud_property:
        params["filter"] = f"{cloud_property} < {cloud_cover_max}"

    url = f"https://earthengine.googleapis.com/v1alpha/{collection_name}:listImages?{urllib.parse.urlencode(params)}"

    # Check cache first
    cache_key = _cache_key("search", collection, bbox, start_date, end_date, cloud_cover_max)
    cached_data = _get_cached(cache_key)

    if cached_data:
        data = json.loads(cached_data.decode("utf-8"))
        images = data.get("images", [])
    else:
        # Fetch from API
        print(f"[INFO] Searching images in Earth Engine: {collection_id}")
        response = session.get(url)

        if response.status_code != 200:
            raise RuntimeError(
                f"Earth Engine API error: {response.status_code}\n{response.text[:500]}"
            )

        data = response.json()
        images = data.get("images", [])

        # Cache the result (1 day TTL for search results)
        _set_cached(cache_key, json.dumps(data).encode("utf-8"), ttl=86400)

    # Sort by cloud coverage and limit
    def get_cloud_coverage(img):
        props = img.get("properties", {})
        return props.get(cloud_property, 9999) if cloud_property else 0

    images_sorted = sorted(images, key=get_cloud_coverage)[:limit]

    # Format response
    result = []
    for img in images_sorted:
        props = img.get("properties", {})
        result.append(
            {
                "id": img.get("id", ""),
                "date": img.get("startTime", ""),
                "cloud_coverage": (
                    props.get(cloud_property, None) if cloud_property else None
                ),
                "bands": [band["id"] for band in img.get("bands", [])],
            }
        )

    return result


def _get_image_metadata_internal(image_id: str) -> Dict[str, Any]:
    """Internal function to get image metadata (not exposed as MCP tool)."""
    # Check cache first
    cache_key = _cache_key("metadata", image_id)
    cached_data = _get_cached(cache_key)

    if cached_data:
        return json.loads(cached_data.decode("utf-8"))

    # Fetch from API
    session = _get_session()

    project = "projects/earthengine-public"
    name = f"{project}/assets/{image_id}"
    url = f"https://earthengine.googleapis.com/v1alpha/{name}"

    print(f">> Fetching metadata from Earth Engine API: {image_id}")
    response = session.get(url)

    if response.status_code != 200:
        raise RuntimeError(
            f"Earth Engine API error: {response.status_code}\n{response.text[:500]}"
        )

    result = response.json()

    # Cache the result (7 days TTL for metadata)
    _set_cached(cache_key, json.dumps(result).encode("utf-8"), ttl=604800)

    return result


@dev_mcp_tool(
    mcp,
    description="""
Get detailed metadata for a specific Earth Engine image.

Parameters:
    image_id (str): Full image ID (e.g., "COPERNICUS/S2_SR_HARMONIZED/20230604T082849_20230604T083836_T37UDB")

Returns:
    dict: Image metadata including:
        - id: Image ID
        - date: Acquisition date
        - properties: All image properties
        - bands: Detailed band information (name, data type, grid, CRS, resolution)
        - geometry: Image footprint
        
Example:
    get_image_metadata("COPERNICUS/S2_SR_HARMONIZED/20230604T082849_20230604T083836_T37UDB")
""",
)
def get_image_metadata(image_id: str) -> Dict[str, Any]:
    """Get detailed metadata for an Earth Engine image."""
    return _get_image_metadata_internal(image_id)


def _download_band_internal(
    image_id: str,
    band_name: str,
    bbox: List[float],
    output_path: str,
    resolution: int = 30
) -> str:
    """Internal function to download a single band (not exposed as MCP tool)."""
    # Check if file already exists (file-level cache)
    output_file = TEMP_DIR / output_path
    if output_file.exists():
        print(f"[INFO] File cache hit: {output_file}")
        return f"Result saved at {output_file}"

    session = _get_session()

    # Get image metadata first to check CRS
    metadata = _get_image_metadata_internal(image_id)
    bands = metadata.get("bands", [])

    # Find the band
    band_info = None
    for b in bands:
        if b["id"] == band_name:
            band_info = b
            break

    if not band_info:
        available_bands = [b["id"] for b in bands]
        raise ValueError(
            f"Band '{band_name}' not found. Available bands: {available_bands}"
        )

    # Extract grid info
    grid = band_info.get("grid", {})
    crs_code = grid.get("crsCode", "EPSG:4326")

    # Build bounding box
    min_lon, min_lat, max_lon, max_lat = bbox

    # Calculate dimensions based on resolution
    # Approximate: 1 degree ~ 111 km at equator
    bbox_width_deg = max_lon - min_lon
    bbox_height_deg = max_lat - min_lat
    bbox_width_m = bbox_width_deg * 111000 * np.cos(np.radians((min_lat + max_lat) / 2))
    bbox_height_m = bbox_height_deg * 111000

    width = int(bbox_width_m / resolution)
    height = int(bbox_height_m / resolution)
    
    # Ensure minimum dimensions
    if width < 10:
        width = 10
    if height < 10:
        height = 10

    # Prepare request
    project = "projects/earthengine-public"
    name = f"{project}/assets/{image_id}"
    url = f"https://earthengine.googleapis.com/v1alpha/{name}:getPixels"

    # Use region (GeoJSON polygon) instead of affineTransform for correct geographic bounds
    region = {
        "type": "Polygon",
        "coordinates": [
            [
                [min_lon, min_lat],
                [min_lon, max_lat],
                [max_lon, max_lat],
                [max_lon, min_lat],
                [min_lon, min_lat],
            ]
        ],
    }

    body = json.dumps(
        {
            "fileFormat": "NPY",
            "bandIds": [band_name],
            "region": region,
            "grid": {
                "dimensions": {"width": width, "height": height},
            },
        }
    )

    # Check Redis cache for pixel data
    cache_key = _cache_key("pixels", image_id, band_name, bbox, resolution)
    cached_pixels = _get_cached(cache_key)

    if cached_pixels:
        array = np.load(io.BytesIO(cached_pixels))
    else:
        # Fetch from API
        print(f"[INFO] Downloading band from Earth Engine: {image_id}/{band_name}")
        response = session.post(url, data=body)

        if response.status_code != 200:
            raise RuntimeError(
                f"Earth Engine getPixels error: {response.status_code}\n{response.text[:500]}"
            )

        # Cache pixel data (3 days TTL)
        _set_cached(cache_key, response.content, ttl=259200)

        # Load numpy array
        array = np.load(io.BytesIO(response.content))

    # Handle structured array (multi-band NPY from Earth Engine)
    if array.dtype.names is not None:
        # Structured array - extract the requested band
        if band_name in array.dtype.names:
            array = array[band_name]
        else:
            # Take first field if band name doesn't match
            array = array[array.dtype.names[0]]
    
    # If array has multiple bands (3D), take first
    if array.ndim == 3:
        array = array[:, :, 0]
    
    # Ensure array is 2D and has simple dtype
    if array.ndim != 2:
        raise ValueError(f"Unexpected array shape: {array.shape}")

    # Create output directory
    output_file = TEMP_DIR / output_path
    output_file.parent.mkdir(parents=True, exist_ok=True)

    # Create GeoTIFF
    transform = from_bounds(min_lon, min_lat, max_lon, max_lat, width, height)
    
    # Determine appropriate nodata value based on dtype
    if np.issubdtype(array.dtype, np.unsignedinteger):
        nodata_value = 0  # Use 0 for unsigned integers
    elif np.issubdtype(array.dtype, np.floating):
        nodata_value = -9999.0  # Use -9999 for floats
    else:
        nodata_value = -9999  # Use -9999 for signed integers

    with rasterio.open(
        output_file,
        "w",
        driver="GTiff",
        height=height,
        width=width,
        count=1,
        dtype=array.dtype,
        crs=crs_code,
        transform=transform,
        compress="lzw",
        nodata=nodata_value,
    ) as dst:
        dst.write(array, 1)

    # Create PNG preview
    png_file = output_file.with_suffix('.png')
    try:
        preview_created, preview_backend = _create_png_preview(output_file, png_file, band_name)
        if preview_created and png_file.exists():
            print(f"[INFO] PNG preview created ({preview_backend}): {png_file}")
        else:
            print(f"[WARN] PNG preview skipped: {png_file} ({preview_backend})")
    except Exception as e:
        print(f"[WARN] Could not create PNG preview: {e}")

    return f"Result saved at {output_file}"


@dev_mcp_tool(
    mcp,
    description="""
Download a single band from an Earth Engine image as GeoTIFF.

Choose band_name by semantic role, not by band number alone. Examples:
    - Sentinel-2 B11/B12 = SWIR for built-up/moisture indices, not thermal.
    - Landsat Collection 2 ST_B10 = surface temperature for LST/heat-island tasks.
    - Sentinel-2 B4 = Red visible reflectance; useful for visible turbidity
      proxy tools. Sentinel-2 B3/B8 are for water delineation/NDWI-like tasks,
      not a replacement for turbidity unless the analysis explicitly uses them.
    - NDBI needs SWIR + NIR; NDVI needs NIR + Red.

Parameters:
    image_id (str): Full image ID
    band_name (str): Band name (e.g., "B5" for NIR, "B6" for SWIR1)
    bbox (list[float]): Bounding box [min_lon, min_lat, max_lon, max_lat]
    output_path (str): Relative output path (e.g., "moscow/nir_2021-06-04.tif")
    resolution (int): Spatial resolution in meters (default: 30)

Returns:
    str: "Result saved at {TEMP_DIR}/{output_path}"
    
Example:
    download_band("NASA/HLS/HLSL30/v002/T37UDB_20210604T082849", "B5", 
                  [37.6, 55.7, 37.7, 55.8], "moscow/nir_2021-06-04.tif", 30)
""",
)
def download_band(
    image_id: str,
    band_name: str,
    bbox: List[float],
    output_path: str,
    resolution: int = 30,
) -> str:
    """Download a single band from Earth Engine image as GeoTIFF."""
    return _download_band_internal(image_id, band_name, bbox, output_path, resolution)


@dev_mcp_tool(
    mcp,
    description="""
Download multiple bands from an Earth Engine image as GeoTIFF files (batch operation).

Use this for all bands needed from the same scene/date so downstream index tools
receive aligned inputs. Choose bands by the required physical role:
    - built-up proxy: SWIR + NIR
    - vegetation proxy: NIR + Red
    - optical turbidity proxy: visible reflectance band(s), normally Red
    - thermal analysis: true thermal/LST bands from Landsat Collection 2
Do not download Red + NIR as the main input for built-up area/share.

Parameters:
    image_id (str): Full image ID
    band_names (list[str]): List of band names (e.g., ["B5", "B6"])
    bbox (list[float]): Bounding box [min_lon, min_lat, max_lon, max_lat]
    output_paths (list[str]): List of relative output paths (one per band)
    resolution (int): Spatial resolution in meters (default: 30)

Returns:
    list[str]: List of result messages with full paths
    
Example:
    download_bands("NASA/HLS/HLSL30/v002/T37UDB_20210604T082849", 
                   ["B5", "B6"],
                   [37.6, 55.7, 37.7, 55.8],
                   ["moscow/nir_2021-06-04.tif", "moscow/swir_2021-06-04.tif"],
                   30)
""",
)
def download_bands(
    image_id: str,
    band_names: List[str],
    bbox: List[float],
    output_paths: List[str],
    resolution: int = 30,
) -> List[str]:
    """Download multiple bands from Earth Engine image (batch)."""

    if len(band_names) != len(output_paths):
        raise ValueError(
            f"Number of bands ({len(band_names)}) must match number of output paths ({len(output_paths)})"
        )

    results = []
    for band_name, output_path in zip(band_names, output_paths):
        result = _download_band_internal(
            image_id, band_name, bbox, output_path, resolution
        )
        results.append(result)

    return results


if __name__ == "__main__":
    mcp.run(show_banner=False)
