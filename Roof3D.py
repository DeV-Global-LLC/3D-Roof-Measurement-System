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
from typing import Optional
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
from concurrent.futures import ThreadPoolExecutor

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="3D Roof Measurement System")

# --- Configuration ---
class Config:
    USGS_API_KEY = os.getenv("USGS_API_KEY", "your_usgs_key")
    OPENTOPO_API_KEY = os.getenv("OPENTOPO_API_KEY", "your_opentopo_key")
    GOOGLE_MAPS_API_KEY = os.getenv("GOOGLE_MAPS_API_KEY", None)  # Optional
    MAX_WORKERS = 4
    DEEPROOF_MODEL_WEIGHTS = os.getenv("DEEPROOF_MODEL_WEIGHTS", "model_final.pth")
    DEEPROOF_MODEL_CONFIG = os.getenv("DEEPROOF_MODEL_CONFIG", "COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml")

config = Config()

# --- Initialize DeepRoof Model ---
def initialize_deeproof_model():
    cfg = get_cfg()
    cfg.merge_from_file(model_zoo.get_config_file(config.DEEPROOF_MODEL_CONFIG))
    cfg.MODEL.WEIGHTS = config.DEEPROOF_MODEL_WEIGHTS
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.7  # Confidence threshold
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 1  # Only roof class
    cfg.MODEL.DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    
    predictor = DefaultPredictor(cfg)
    return predictor

# Load model at startup
deeproof_predictor = initialize_deeproof_model()

# --- Models ---
class LocationRequest(BaseModel):
    address: Optional[str] = None
    lat: Optional[float] = None
    lon: Optional[float] = None
    bbox: Optional[list] = None  # [minx, miny, maxx, maxy]

class MeasurementResult(BaseModel):
    area_sqft: float
    area_sqm: float
    slope_degrees: float
    pitch: str
    roof_type: str
    vertices: list
    geometry: dict
    data_source: str
    confidence: float

# --- Helper Functions ---
@tenacity.retry(
    stop=tenacity.stop_after_attempt(3),
    wait=tenacity.wait_exponential(multiplier=1, min=4, max=10),
    before_sleep=tenacity.before_sleep_log(logger, logging.WARNING),
)
def fetch_usgs_naip(bbox: list, year: int = 2023) -> str:
    """Fetch NAIP imagery from USGS EarthExplorer API"""
    url = "https://tnmaccess.nationalmap.gov/api/v1/products"
    params = {
        "datasets": "National Agriculture Imagery Program (NAIP)",
        "bbox": f"{bbox[0]},{bbox[1]},{bbox[2]},{bbox[3]}",
        "startDate": f"{year}-01-01",
        "endDate": f"{year}-12-31",
        "prodFormats": "GeoTIFF",
        "api_key": config.USGS_API_KEY,
    }
    
    response = requests.get(url, params=params)
    response.raise_for_status()
    data = response.json()
    
    if not data.get("items"):
        raise ValueError("No NAIP imagery found for this area")
    
    download_url = data["items"][0]["downloadURL"]
    with tempfile.NamedTemporaryFile(suffix=".tif", delete=False) as tmp:
        img_response = requests.get(download_url, stream=True)
        img_response.raise_for_status()
        for chunk in img_response.iter_content(chunk_size=8192):
            tmp.write(chunk)
        return tmp.name

@tenacity.retry(
    stop=tenacity.stop_after_attempt(3),
    wait=tenacity.wait_exponential(multiplier=1, min=4, max=10),
)
def fetch_lidar(bbox: list) -> str:
    """Fetch LiDAR data from OpenTopography"""
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
    
    response = requests.get(url, params=params)
    response.raise_for_status()
    
    with tempfile.NamedTemporaryFile(suffix=".laz", delete=False) as tmp:
        tmp.write(response.content)
        return tmp.name

def fetch_google_maps(bbox: list, zoom: int = 20) -> str:
    """Fetch satellite imagery from Google Maps (paid)"""
    if not config.GOOGLE_MAPS_API_KEY:
        raise ValueError("Google Maps API key not configured")
    
    center = f"{(bbox[1]+bbox[3])/2},{(bbox[0]+bbox[2])/2}"
    size = "640x640"
    url = f"https://maps.googleapis.com/maps/api/staticmap?center={center}&zoom={zoom}&size={size}&maptype=satellite&key={config.GOOGLE_MAPS_API_KEY}"
    
    response = requests.get(url)
    response.raise_for_status()
    
    with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp:
        tmp.write(response.content)
        return tmp.name

def classify_roof_type(slope_degrees: float) -> str:
    """Classify roof based on slope"""
    if slope_degrees < 5:
        return "Flat"
    elif 5 <= slope_degrees < 15:
        return "Low Slope"
    elif 15 <= slope_degrees < 30:
        return "Medium Slope"
    elif 30 <= slope_degrees < 45:
        return "Steep Slope"
    else:
        return "Very Steep Slope"

# --- Core Processing ---
def detect_roof(image_path: str) -> tuple[Polygon, float]:
    """Detect roof polygons using DeepRoof model"""
    try:
        # Read and preprocess image
        img = cv2.imread(image_path)
        if img is None:
            raise ValueError("Could not read image file")
        
        # Run DeepRoof prediction
        outputs = deeproof_predictor(img)
        
        # Get the best roof prediction (highest score)
        instances = outputs["instances"]
        if len(instances) == 0:
            raise ValueError("No roofs detected in the image")
        
        # Get the highest confidence prediction
        scores = instances.scores.cpu().numpy()
        best_idx = np.argmax(scores)
        confidence = scores[best_idx]
        
        # Get the mask polygon
        mask = instances.pred_masks[best_idx].cpu().numpy()
        mask = mask.astype(np.uint8) * 255
        
        # Find contours from mask
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            raise ValueError("Could not extract roof contour from mask")
        
        # Simplify the contour to a polygon
        largest_contour = max(contours, key=cv2.contourArea)
        epsilon = 0.01 * cv2.arcLength(largest_contour, True)
        approx = cv2.approxPolyDP(largest_contour, epsilon, True)
        
        # Create Shapely polygon
        polygon = Polygon([(point[0][0], point[0][1]) for point in approx])
        
        # Simplify the polygon if too complex
        if len(polygon.exterior.coords) > 100:
            polygon = polygon.simplify(5.0, preserve_topology=True)
        
        return polygon, float(confidence)
    
    except Exception as e:
        logger.error(f"Roof detection failed: {str(e)}")
        raise

def calculate_roof_metrics(roof_polygon: Polygon, laz_file: str) -> dict:
    """Calculate 3D roof metrics from LiDAR"""
    try:
        pipeline = {
            "pipeline": [
                {"type": "readers.las", "filename": laz_file},
                {"type": "filters.crop", "polygon": roof_polygon.wkt},
                {"type": "filters.range", "limits": "Classification[6:6]"},  # Building points
                {"type": "filters.estimaterank", "knn": 8},
                {"type": "filters.normal"},
            ]
        }
        
        pipeline = pdal.Pipeline(json.dumps(pipeline))
        pipeline.execute()
        
        if len(pipeline.arrays) == 0:
            raise ValueError("No LiDAR points found within roof polygon")
        
        arr = pipeline.arrays[0]
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(np.c_[arr["X"], arr["Y"], arr["Z"]])
        
        # Calculate slope using RANSAC plane fitting
        plane_model, inliers = pcd.segment_plane(0.1, 3, 1000)
        a, b, c, d = plane_model
        slope_deg = np.degrees(np.arccos(c / np.sqrt(a**2 + b**2 + c**2)))
        
        # Calculate area (convert from pixels to m²)
        with rasterio.open(laz_file.replace(".laz", ".tif")) as src:
            transform = src.transform
            pixel_area = transform[0] * transform[0]  # m² per pixel
            area_m2 = roof_polygon.area * pixel_area
        
        return {
            "slope_degrees": slope_deg,
            "area_m2": area_m2,
            "plane_model": plane_model,
        }
    
    except Exception as e:
        logger.error(f"Roof metrics calculation failed: {str(e)}")
        raise

# --- API Endpoints ---
@app.post("/measure", response_model=MeasurementResult)
async def measure_roof(
    location: LocationRequest, 
    background_tasks: BackgroundTasks
):
    try:
        # Step 1: Get bounding box
        if location.bbox:
            bbox = location.bbox
        elif location.lat and location.lon:
            bbox = [
                location.lon - 0.0005,  # ~50m buffer
                location.lat - 0.0005,
                location.lon + 0.0005,
                location.lat + 0.0005,
            ]
        else:
            raise HTTPException(400, "Address geocoding not implemented")

        # Step 2: Fetch data (parallel)
        with ThreadPoolExecutor(max_workers=config.MAX_WORKERS) as executor:
            img_future = executor.submit(fetch_usgs_naip, bbox)
            lidar_future = executor.submit(fetch_lidar, bbox)
            image_path = img_future.result()
            laz_path = lidar_future.result()

        # Step 3: Process
        roof_polygon, confidence = detect_roof(image_path)
        metrics = calculate_roof_metrics(roof_polygon, laz_path)
        
        # Step 4: Cleanup (async)
        background_tasks.add_task(os.unlink, image_path)
        background_tasks.add_task(os.unlink, laz_path)

        return {
            "area_sqft": metrics["area_m2"] * 10.764,
            "area_sqm": metrics["area_m2"],
            "slope_degrees": metrics["slope_degrees"],
            "pitch": f"1:{round(1/np.tan(np.radians(metrics['slope_degrees'])), 2)}",
            "roof_type": classify_roof_type(metrics["slope_degrees"]),
            "vertices": list(roof_polygon.exterior.coords),
            "geometry": gpd.GeoSeries([roof_polygon]).__geo_interface__,
            "data_source": "USGS NAIP + 3DEP LiDAR",
            "confidence": confidence,
        }

    except Exception as e:
        logger.error(f"Error processing request: {str(e)}")
        raise HTTPException(500, str(e))

# --- Run ---
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
