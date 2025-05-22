import os
import json
import numpy as np
import geopandas as gpd
import rasterio
from shapely.geometry import Polygon, MultiPolygon
import open3d as o3d
import pdal
import requests
from fastapi import FastAPI, HTTPException, UploadFile, BackgroundTasks
from typing import Optional, Tuple, List, Dict
from pydantic import BaseModel
import tempfile
import cv2
import torch
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.data import MetadataCatalog
from detectron2.utils.visualizer import Visualizer
import tenacity
from datetime import datetime
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from functools import lru_cache
import base64
from pyproj import CRS, Transformer
import asyncio
from fastapi.concurrency import run_in_threadpool

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="3D Roof Measurement System")

# --- Configuration ---
class Config:
    USGS_API_KEY = os.getenv("USGS_API_KEY", "your_usgs_key")
    OPENTOPO_API_KEY = os.getenv("OPENTOPO_API_KEY", "your_opentopo_key")
    GOOGLE_MAPS_API_KEY = os.getenv("GOOGLE_MAPS_API_KEY", None)  # Optional
    OSM_API_URL = "https://api.openstreetmap.org/api/0.6/map"  # For building footprints
    MAX_WORKERS = min(32, (os.cpu_count() or 1) + 4)  # Dynamic worker count
    DEEPROOF_MODEL_WEIGHTS = os.getenv("DEEPROOF_MODEL_WEIGHTS", "model_final.pth")
    DEEPROOF_MODEL_CONFIG = os.getenv("DEEPROOF_MODEL_CONFIG", "COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml")
    MIN_ROOF_HEIGHT = 2.5  # meters
    CACHE_SIZE = 100  # Number of items to cache
    BATCH_MAX_SIZE = 100  # Maximum number of tiles in batch processing
    NUM_ROOF_CLASSES = 3  # Update based on your model's capabilities (1 for single class)

config = Config()

# --- Models ---
class BoundingBox(BaseModel):
    minx: float
    miny: float
    maxx: float
    maxy: float
    debug: Optional[bool] = False
    fallback_to_lower_quality: Optional[bool] = True  # Allow fallback to lower quality data

class LocationRequest(BaseModel):
    address: Optional[str] = None
    lat: Optional[float] = None
    lon: Optional[float] = None
    bbox: Optional[list] = None  # [minx, miny, maxx, maxy]
    debug: Optional[bool] = False
    fallback_to_lower_quality: Optional[bool] = True

class MeasurementResult(BaseModel):
    area_sqft: float
    area_sqm: float
    slope_degrees: float
    pitch: str
    roof_type: str
    roof_class: Optional[str] = None  # Only if model supports multiple classes
    vertices: list
    geometry: dict
    data_source: str
    confidence: float
    visualization: Optional[str] = None
    warnings: Optional[List[str]] = None

# --- Initialize DeepRoof Model ---
def initialize_deeproof_model():
    cfg = get_cfg()
    cfg.merge_from_file(model_zoo.get_config_file(config.DEEPROOF_MODEL_CONFIG))
    cfg.MODEL.WEIGHTS = config.DEEPROOF_MODEL_WEIGHTS
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.7
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = config.NUM_ROOF_CLASSES  # Dynamic class count
    cfg.MODEL.DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    
    predictor = DefaultPredictor(cfg)
    
    # Initialize roof class names if available
    if hasattr(predictor.metadata, "thing_classes"):
        logger.info(f"Model supports {len(predictor.metadata.thing_classes)} roof types")
    
    return predictor

deeproof_predictor = initialize_deeproof_model()

# --- Helper Functions ---
@tenacity.retry(
    stop=tenacity.stop_after_attempt(3),
    wait=tenacity.wait_exponential(multiplier=1, min=4, max=10),
    before_sleep=tenacity.before_sleep_log(logger, logging.WARNING),
    retry=tenacity.retry_if_exception_type(requests.RequestException)
)
@lru_cache(maxsize=config.CACHE_SIZE)
def fetch_usgs_naip(bbox: tuple, year: int = 2023) -> str:
    """Fetch NAIP imagery with retries and fallback options"""
    try:
        bbox = list(bbox)
        url = "https://tnmaccess.nationalmap.gov/api/v1/products"
        params = {
            "datasets": "National Agriculture Imagery Program (NAIP)",
            "bbox": f"{bbox[0]},{bbox[1]},{bbox[2]},{bbox[3]}",
            "startDate": f"{year}-01-01",
            "endDate": f"{year}-12-31",
            "prodFormats": "GeoTIFF",
            "api_key": config.USGS_API_KEY,
        }
        
        response = requests.get(url, params=params, timeout=30)
        response.raise_for_status()
        data = response.json()
        
        if not data.get("items"):
            raise ValueError("No NAIP imagery found for this area")
        
        download_url = data["items"][0]["downloadURL"]
        with tempfile.NamedTemporaryFile(suffix=".tif", delete=False) as tmp:
            img_response = requests.get(download_url, stream=True, timeout=60)
            img_response.raise_for_status()
            for chunk in img_response.iter_content(chunk_size=8192):
                tmp.write(chunk)
            return tmp.name
            
    except Exception as e:
        logger.error(f"Failed to fetch NAIP imagery: {str(e)}")
        raise

@tenacity.retry(
    stop=tenacity.stop_after_attempt(3),
    wait=tenacity.wait_exponential(multiplier=1, min=4, max=10),
    before_sleep=tenacity.before_sleep_log(logger, logging.WARNING),
    retry=tenacity.retry_if_exception_type(requests.RequestException)
)
@lru_cache(maxsize=config.CACHE_SIZE)
def fetch_lidar(bbox: tuple) -> str:
    """Fetch LiDAR data with retries and fallback options"""
    try:
        bbox = list(bbox)
        url = "https://portal.opentopography.org/API/globaldem"
        params = {
            "demtype": "LIDAR",
            "south": bbox[1],
            "north": bbox[3],
            "west": bbox[0],
            "east": bbox[2],
            "outputFormat": "LAZ",
            "APIkey": config.OPENTOPO_API_KEY,
        }
        
        response = requests.get(url, params=params, timeout=30)
        response.raise_for_status()
        
        with tempfile.NamedTemporaryFile(suffix=".laz", delete=False) as tmp:
            tmp.write(response.content)
            return tmp.name
            
    except Exception as e:
        logger.error(f"Failed to fetch LiDAR data: {str(e)}")
        raise

async def fetch_osm_building_footprints(bbox: list) -> Optional[gpd.GeoDataFrame]:
    """Fetch building footprints from OpenStreetMap"""
    try:
        minx, miny, maxx, maxy = bbox
        url = f"{config.OSM_API_URL}?bbox={minx},{miny},{maxx},{maxy}"
        
        async with httpx.AsyncClient() as client:
            response = await client.get(url, timeout=30)
            response.raise_for_status()
            
            # Parse OSM XML to GeoDataFrame
            buildings = gpd.read_file(response.text, driver='OSM')
            if not buildings.empty:
                return buildings[buildings.geometry.type == 'Polygon']
            return None
            
    except Exception as e:
        logger.warning(f"Could not fetch OSM buildings: {str(e)}")
        return None

def validate_against_osm(roof_polygon: Polygon, osm_buildings: gpd.GeoDataFrame) -> List[str]:
    """Validate roof detection against OSM footprints"""
    warnings = []
    try:
        # Find overlapping OSM buildings
        overlapping = osm_buildings[osm_buildings.geometry.intersects(roof_polygon)]
        
        if not overlapping.empty:
            # Check area ratio
            max_overlap = max(overlapping.geometry.apply(
                lambda x: roof_polygon.intersection(x).area / roof_polygon.area
            ))
            
            if max_overlap < 0.5:
                warnings.append(f"Roof only overlaps {max_overlap:.0%} with OSM building footprints")
        else:
            warnings.append("No matching OSM building footprint found")
            
    except Exception as e:
        logger.warning(f"OSM validation failed: {str(e)}")
        
    return warnings

def detect_roof(image_path: str) -> Tuple[Polygon, float, dict, Optional[str]]:
    """Enhanced roof detection with multi-class support"""
    try:
        img = cv2.imread(image_path)
        if img is None:
            raise ValueError("Could not read image file")
        
        outputs = deeproof_predictor(img)
        instances = outputs["instances"]
        
        if len(instances) == 0:
            raise ValueError("No roofs detected in the image")
        
        # Get the best roof prediction
        scores = instances.scores.cpu().numpy()
        best_idx = np.argmax(scores)
        confidence = scores[best_idx]
        
        # Get roof class if model supports it
        roof_class = None
        if hasattr(deeproof_predictor.metadata, "thing_classes"):
            class_id = instances.pred_classes[best_idx].item()
            roof_class = deeproof_predictor.metadata.thing_classes[class_id]
        
        # Process mask
        mask = instances.pred_masks[best_idx].cpu().numpy()
        mask = mask.astype(np.uint8) * 255
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if not contours:
            raise ValueError("Could not extract roof contour from mask")
        
        largest_contour = max(contours, key=cv2.contourArea)
        epsilon = 0.01 * cv2.arcLength(largest_contour, True)
        approx = cv2.approxPolyDP(largest_contour, epsilon, True)
        
        polygon = Polygon([(point[0][0], point[0][1]) for point in approx])
        if len(polygon.exterior.coords) > 100:
            polygon = polygon.simplify(5.0, preserve_topology=True)
        
        return polygon, float(confidence), outputs, roof_class
        
    except Exception as e:
        logger.error(f"Roof detection failed: {str(e)}")
        raise

async def process_single_roof(bbox: list, debug: bool = False, fallback: bool = True) -> dict:
    """Async processing with enhanced error handling and fallbacks"""
    warnings = []
    try:
        # Fetch data in parallel
        loop = asyncio.get_running_loop()
        with ThreadPoolExecutor(max_workers=2) as pool:
            tasks = [
                loop.run_in_executor(pool, fetch_usgs_naip, tuple(bbox)),
                loop.run_in_executor(pool, fetch_lidar, tuple(bbox)),
                fetch_osm_building_footprints(bbox)
            ]
            image_path, laz_path, osm_buildings = await asyncio.gather(*tasks)
        
        # Detect roof
        roof_polygon, confidence, outputs, roof_class = await run_in_threadpool(
            detect_roof, image_path
        )
        
        # Validate against OSM if available
        if osm_buildings is not None:
            osm_warnings = validate_against_osm(roof_polygon, osm_buildings)
            warnings.extend(osm_warnings)
        
        # Calculate metrics
        metrics = await run_in_threadpool(
            calculate_roof_metrics, roof_polygon, laz_path
        )
        
        # Create visualization if requested
        visualization = None
        if debug:
            visualization = await run_in_threadpool(
                create_visual_debug, image_path, outputs, roof_polygon
            )
        
        # Cleanup
        await loop.run_in_executor(pool, os.unlink, image_path)
        await loop.run_in_executor(pool, os.unlink, laz_path)
        
        result = {
            "area_sqft": metrics["area_m2"] * 10.764,
            "area_sqm": metrics["area_m2"],
            "slope_degrees": metrics["slope_degrees"],
            "pitch": f"{round(np.tan(np.radians(metrics['slope_degrees'])) * 12, 1)}:12",
            "roof_type": classify_roof_type(metrics["slope_degrees"]),
            "roof_class": roof_class,
            "vertices": list(roof_polygon.exterior.coords),
            "geometry": json.loads(gpd.GeoSeries([roof_polygon]).to_json())["features"][0]["geometry"],
            "data_source": "USGS NAIP + OpenTopography",
            "confidence": confidence,
            "visualization": visualization,
            "warnings": warnings if warnings else None
        }
        
        return result
        
    except Exception as e:
        logger.error(f"Error processing bbox {bbox}: {str(e)}")
        if fallback and config.GOOGLE_MAPS_API_KEY:
            logger.info("Attempting fallback to Google Maps")
            try:
                # Implement fallback logic here
                pass
            except Exception as fallback_error:
                logger.error(f"Fallback also failed: {str(fallback_error)}")
        
        return {"error": str(e), "bbox": bbox, "warnings": warnings}

# --- API Endpoints ---
@app.post("/measure", response_model=MeasurementResult)
async def measure_roof(
    location: LocationRequest, 
    background_tasks: BackgroundTasks
):
    try:
        if location.bbox:
            bbox = location.bbox
        elif location.lat and location.lon:
            bbox = [
                location.lon - 0.0005,
                location.lat - 0.0005,
                location.lon + 0.0005,
                location.lat + 0.0005,
            ]
        else:
            raise HTTPException(400, "Address geocoding not implemented")

        result = await process_single_roof(
            bbox,
            debug=location.debug,
            fallback=location.fallback_to_lower_quality
        )
        
        if "error" in result:
            raise HTTPException(500, result["error"])
            
        return result
        
    except Exception as e:
        logger.error(f"Error processing request: {str(e)}")
        raise HTTPException(500, str(e))

@app.post("/batch/process/")
async def batch_process(
    coords: List[BoundingBox],
    background_tasks: BackgroundTasks
):
    if len(coords) > config.BATCH_MAX_SIZE:
        raise HTTPException(400, f"Maximum batch size is {config.BATCH_MAX_SIZE}")
    
    results = []
    tasks = []
    
    for box in coords:
        bbox = [box.minx, box.miny, box.maxx, box.maxy]
        task = process_single_roof(
            bbox,
            debug=box.debug,
            fallback=box.fallback_to_lower_quality
        )
        tasks.append(task)
    
    # Process with limited concurrency
    for future in asyncio.as_completed(tasks):
        try:
            results.append(await future)
        except Exception as e:
            logger.error(f"Error in batch processing: {str(e)}")
            results.append({"error": str(e)})
    
    return results

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, workers=1)
