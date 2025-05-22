import os
import json
import numpy as np
import geopandas as gpd
import rasterio
from shapely.geometry import Polygon, MultiPolygon, box, Point
import open3d as o3d
import pdal
import requests
from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect, BackgroundTasks
from typing import Optional, Tuple, List, Dict, AsyncGenerator
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
import logging
from concurrent.futures import ThreadPoolExecutor
from functools import lru_cache
import base64
from pyproj import CRS, Transformer
import asyncio
from fastapi.concurrency import run_in_threadpool
import httpx
import gc
import laspy
from sklearn.linear_model import LinearRegression
from scipy.spatial import ConvexHull
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import osmnx as ox

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="3D Roof Measurement System")

# --- Configuration ---
class Config:
    USGS_API_KEY = os.getenv("USGS_API_KEY", "your_usgs_key")
    OPENTOPO_API_KEY = os.getenv("OPENTOPO_API_KEY", "your_opentopo_key")
    GOOGLE_MAPS_API_KEY = os.getenv("GOOGLE_MAPS_API_KEY", None)
    OSM_API_URL = "https://api.openstreetmap.org/api/0.6/map"
    MAX_WORKERS = min(32, (os.cpu_count() or 1) + 4)
    DEEPROOF_MODEL_WEIGHTS = os.getenv("DEEPROOF_MODEL_WEIGHTS", "model_final.pth")
    DEEPROOF_MODEL_CONFIG = os.getenv("DEEPROOF_MODEL_CONFIG", "COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml")
    MIN_ROOF_HEIGHT = 2.5
    CACHE_SIZE = 100
    BATCH_MAX_SIZE = 100
    NUM_ROOF_CLASSES = 3
    USE_GPU = torch.cuda.is_available()  # Automatically detect GPU
    CHUNK_SIZE = 1024 * 1024  # 1MB chunks for streaming
    OPENTOPO_API_URL = "https://portal.opentopography.org/API/globaldem"

config = Config()

# --- Models ---
class BoundingBox(BaseModel):
    minx: float
    miny: float
    maxx: float
    maxy: float
    debug: Optional[bool] = False
    fallback_to_lower_quality: Optional[bool] = True

class LocationRequest(BaseModel):
    address: Optional[str] = None
    lat: Optional[float] = None
    lon: Optional[float] = None
    bbox: Optional[list] = None
    debug: Optional[bool] = False
    fallback_to_lower_quality: Optional[bool] = True

class MeasurementResult(BaseModel):
    area_sqft: float
    area_sqm: float
    slope_degrees: float
    pitch: str
    roof_type: str
    roof_class: Optional[str] = None
    vertices: list
    geometry: dict
    data_source: str
    confidence: float
    visualization: Optional[str] = None
    warnings: Optional[List[str]] = None

class ProgressUpdate(BaseModel):
    stage: str
    progress: float
    message: str

# --- Initialize DeepRoof Model with GPU support ---
def initialize_deeproof_model():
    cfg = get_cfg()
    cfg.merge_from_file(model_zoo.get_config_file(config.DEEPROOF_MODEL_CONFIG))
    cfg.MODEL.WEIGHTS = config.DEEPROOF_MODEL_WEIGHTS
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.7
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = config.NUM_ROOF_CLASSES
    
    if config.USE_GPU:
        cfg.MODEL.DEVICE = "cuda"
        # Optimize for GPU memory
        torch.backends.cudnn.benchmark = True
        torch.cuda.empty_cache()
    else:
        cfg.MODEL.DEVICE = "cpu"
        logger.warning("Running on CPU - performance will be degraded")
    
    predictor = DefaultPredictor(cfg)
    
    if hasattr(predictor.metadata, "thing_classes"):
        logger.info(f"Model supports {len(predictor.metadata.thing_classes)} roof types")
    
    return predictor

deeproof_predictor = initialize_deeproof_model()

# --- WebSocket Manager ---
class ConnectionManager:
    def __init__(self):
        self.active_connections: List[WebSocket] = []

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)

    def disconnect(self, websocket: WebSocket):
        self.active_connections.remove(websocket)

    async def send_progress(self, message: ProgressUpdate, websocket: WebSocket):
        await websocket.send_json(message.dict())

manager = ConnectionManager()

# --- OpenTopography LiDAR Fetching ---
@tenacity.retry(
    stop=tenacity.stop_after_attempt(3),
    wait=tenacity.wait_exponential(multiplier=1, min=4, max=10),
    before_sleep=tenacity.before_sleep_log(logger, logging.WARNING),
    retry=tenacity.retry_if_exception_type(requests.RequestException)
)
async def fetch_lidar(bounds: Tuple[float, float, float, float], websocket: WebSocket = None) -> str:
    """Fetch LiDAR data from OpenTopography API with progress streaming"""
    try:
        min_lon, min_lat, max_lon, max_lat = bounds
        params = {
            "demtype": "COP30",
            "south": min_lat,
            "north": max_lat,
            "west": min_lon,
            "east": max_lon,
            "outputFormat": "LAZ",
            "API_Key": config.OPENTOPO_API_KEY,
        }
        
        # Create temp file
        temp_file = tempfile.NamedTemporaryFile(suffix=".laz", delete=False)
        temp_path = temp_file.name
        temp_file.close()
        
        # Stream download with progress
        async with httpx.AsyncClient() as client:
            async with client.stream(
                "GET",
                config.OPENTOPO_API_URL,
                params=params,
                timeout=30
            ) as response:
                response.raise_for_status()
                total = int(response.headers.get('content-length', 0))
                downloaded = 0
                
                with open(temp_path, 'wb') as f:
                    async for chunk in response.aiter_bytes(chunk_size=config.CHUNK_SIZE):
                        f.write(chunk)
                        downloaded += len(chunk)
                        if websocket:
                            progress = min(100, (downloaded / total) * 100) if total > 0 else 0
                            await manager.send_progress(
                                ProgressUpdate(
                                    stage="lidar_download",
                                    progress=progress,
                                    message=f"Downloading LiDAR data {downloaded/1024/1024:.1f}MB"
                                ),
                                websocket
                            )
        
        logger.info(f"LiDAR data saved to {temp_path}")
        return temp_path
        
    except Exception as e:
        logger.error(f"Failed to fetch LiDAR data: {str(e)}")
        raise HTTPException(500, f"LiDAR fetch failed: {str(e)}")

# --- OSM Building Footprints ---
async def fetch_osm_building_footprints(bounds: Tuple[float, float, float, float]) -> gpd.GeoDataFrame:
    """Fetch OSM building footprints asynchronously"""
    try:
        min_lon, min_lat, max_lon, max_lat = bounds
        geom = box(min_lon, min_lat, max_lon, max_lat)
        
        # Run synchronous OSMNX code in threadpool
        def _fetch_sync():
            gdf = ox.geometries.geometries_from_bbox(
                north=max_lat,
                south=min_lat,
                east=max_lon,
                west=min_lon,
                tags={'building': True}
            )
            buildings = gdf[gdf.geometry.type.isin(["Polygon", "MultiPolygon"])]
            return buildings.reset_index(drop=True)
        
        return await run_in_threadpool(_fetch_sync)
        
    except Exception as e:
        logger.error(f"Failed to fetch OSM buildings: {str(e)}")
        raise HTTPException(500, f"OSM fetch failed: {str(e)}")

# --- Roof Detection from LiDAR ---
def detect_roof(lidar_path: str, building_polygon: Polygon) -> np.ndarray:
    """Extract roof points from LiDAR point cloud inside a building polygon"""
    try:
        with laspy.open(lidar_path) as fh:
            las = fh.read()

        coords = np.vstack((las.x, las.y, las.z)).T
        in_building = np.array([building_polygon.contains(Point(x, y)) for x, y, _ in coords])
        building_points = coords[in_building]
        
        # Filter by height (optional): get top 10% to get roof
        z_threshold = np.percentile(building_points[:, 2], 90)
        roof_points = building_points[building_points[:, 2] > z_threshold]
        
        return roof_points
        
    except Exception as e:
        logger.error(f"Roof detection failed: {str(e)}")
        raise HTTPException(500, f"Roof detection failed: {str(e)}")

# --- Roof Metrics Calculation ---
def calculate_roof_metrics(roof_points: np.ndarray) -> dict:
    """Calculate geometric metrics of a roof"""
    try:
        if len(roof_points) < 3:
            return {}

        x, y, z = roof_points[:, 0], roof_points[:, 1], roof_points[:, 2]

        # Fit plane: z = ax + by + c
        reg = LinearRegression().fit(np.c_[x, y], z)
        a, b = reg.coef_

        slope_rad = np.arctan(np.sqrt(a**2 + b**2))
        slope_deg = np.degrees(slope_rad)

        elevation_range = np.max(z) - np.min(z)

        # Roof area (approx): convex hull projected to XY
        hull = ConvexHull(roof_points[:, :2])
        area = hull.volume  # in 2D, volume is area

        return {
            "slope_degrees": round(slope_deg, 2),
            "elevation_range": round(elevation_range, 2),
            "area_m2": round(area, 2),
        }
        
    except Exception as e:
        logger.error(f"Metrics calculation failed: {str(e)}")
        raise HTTPException(500, f"Metrics calculation failed: {str(e)}")

# --- Roof Type Classification ---
def classify_roof_type(slope_degrees: float) -> str:
    """Classify roof shape based on slope"""
    if slope_degrees < 5:
        return "Flat"
    elif 5 <= slope_degrees < 20:
        return "Low-slope (Shed or Gable)"
    elif slope_degrees >= 20:
        return "Steep-slope (Hip, Gable or Complex)"
    else:
        return "Unknown"

# --- Debug Visualization ---
def create_visual_debug(building_polygon: Polygon, roof_points: np.ndarray) -> str:
    """Generate 3D visualization of roof points and building footprint"""
    try:
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')

        x, y, z = roof_points[:, 0], roof_points[:, 1], roof_points[:, 2]
        ax.scatter(x, y, z, c=z, cmap='viridis', s=1)

        # Building footprint outline
        bx, by = building_polygon.exterior.xy
        bz = [min(z)] * len(bx)
        ax.plot(bx, by, bz, color='red', linewidth=2)

        ax.set_title("Roof Detection Debug")
        
        # Save to temp file and return as base64
        temp_file = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
        plt.savefig(temp_file.name)
        plt.close()
        
        with open(temp_file.name, "rb") as f:
            img_bytes = f.read()
        os.unlink(temp_file.name)
        
        return base64.b64encode(img_bytes).decode('utf-8')
        
    except Exception as e:
        logger.error(f"Visualization failed: {str(e)}")
        return None

# --- Helper Functions with Streaming Support ---
async def stream_file(url: str, websocket: WebSocket = None) -> AsyncGenerator[bytes, None]:
    async with httpx.AsyncClient() as client:
        async with client.stream("GET", url) as response:
            response.raise_for_status()
            total = int(response.headers.get('content-length', 0))
            downloaded = 0
            
            async for chunk in response.aiter_bytes(chunk_size=config.CHUNK_SIZE):
                downloaded += len(chunk)
                if websocket:
                    progress = min(100, (downloaded / total) * 100) if total > 0 else 0
                    await manager.send_progress(
                        ProgressUpdate(
                            stage="download",
                            progress=progress,
                            message=f"Downloading {downloaded/1024/1024:.1f}MB"
                        ),
                        websocket
                    )
                yield chunk

@tenacity.retry(
    stop=tenacity.stop_after_attempt(3),
    wait=tenacity.wait_exponential(multiplier=1, min=4, max=10),
    before_sleep=tenacity.before_sleep_log(logger, logging.WARNING),
    retry=tenacity.retry_if_exception_type(requests.RequestException)
)
@lru_cache(maxsize=config.CACHE_SIZE)
async def fetch_usgs_naip(bbox: tuple, year: int = 2023, websocket: WebSocket = None) -> str:
    """Fetch NAIP imagery with progress streaming"""
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
        
        async with httpx.AsyncClient() as client:
            response = await client.get(url, params=params, timeout=30)
            response.raise_for_status()
            data = response.json()
            
            if not data.get("items"):
                raise ValueError("No NAIP imagery found for this area")
            
            download_url = data["items"][0]["downloadURL"]
            temp_file = tempfile.NamedTemporaryFile(suffix=".tif", delete=False)
            
            async for chunk in stream_file(download_url, websocket):
                temp_file.write(chunk)
            
            temp_file.close()
            return temp_file.name
            
    except Exception as e:
        logger.error(f"Failed to fetch NAIP imagery: {str(e)}")
        raise

async def process_single_roof_streaming(
    bbox: list, 
    debug: bool = False, 
    fallback: bool = True,
    websocket: WebSocket = None
) -> AsyncGenerator[Dict, None]:
    """Stream processing with progress updates"""
    try:
        # Stage 1: Data Fetching
        if websocket:
            await manager.send_progress(
                ProgressUpdate(
                    stage="fetching",
                    progress=0,
                    message="Starting data download"
                ),
                websocket
            )
        
        # Fetch data in parallel with progress
        image_path, laz_path, osm_buildings = await asyncio.gather(
            fetch_usgs_naip(tuple(bbox), websocket=websocket),
            fetch_lidar(tuple(bbox), websocket=websocket),
            fetch_osm_building_footprints(bbox)
        )
        
        # Process each building
        for idx, building in osm_buildings.iterrows():
            building_polygon = building.geometry
            
            if websocket:
                await manager.send_progress(
                    ProgressUpdate(
                        stage="detection",
                        progress=30,
                        message=f"Processing building {idx+1}/{len(osm_buildings)}"
                    ),
                    websocket
                )
            
            # Stage 2: Roof Detection
            roof_points = await run_in_threadpool(
                detect_roof, laz_path, building_polygon
            )
            
            if websocket:
                await manager.send_progress(
                    ProgressUpdate(
                        stage="metrics",
                        progress=60,
                        message="Calculating roof metrics"
                    ),
                    websocket
                )
            
            # Stage 3: Metrics Calculation
            metrics = await run_in_threadpool(
                calculate_roof_metrics, roof_points
            )
            
            # Stage 4: Classification
            roof_type = classify_roof_type(metrics.get("slope_degrees", 0))
            
            # Stage 5: Visualization (if requested)
            visualization = None
            if debug:
                if websocket:
                    await manager.send_progress(
                        ProgressUpdate(
                            stage="visualization",
                            progress=80,
                            message="Generating debug visualization"
                        ),
                        websocket
                    )
                visualization = await run_in_threadpool(
                    create_visual_debug, building_polygon, roof_points
                )
            
            # Final result
            result = {
                "area_sqft": metrics.get("area_m2", 0) * 10.764,
                "area_sqm": metrics.get("area_m2", 0),
                "slope_degrees": metrics.get("slope_degrees", 0),
                "pitch": f"{round(np.tan(np.radians(metrics.get('slope_degrees', 0))) * 12, 1)}:12",
                "roof_type": roof_type,
                "vertices": list(building_polygon.exterior.coords),
                "geometry": json.loads(gpd.GeoSeries([building_polygon]).to_json())["features"][0]["geometry"],
                "data_source": "USGS NAIP + OpenTopography + OSM",
                "confidence": 1.0,  # Placeholder
                "visualization": visualization,
                "warnings": []
            }
            
            yield result
        
        # Cleanup
        await asyncio.gather(
            run_in_threadpool(os.unlink, image_path),
            run_in_threadpool(os.unlink, laz_path)
        )
        
    except Exception as e:
        logger.error(f"Error processing bbox {bbox}: {str(e)}")
        yield {"error": str(e), "bbox": bbox}

# --- WebSocket Endpoint ---
@app.websocket("/ws/measure")
async def websocket_measure(websocket: WebSocket):
    await manager.connect(websocket)
    try:
        while True:
            data = await websocket.receive_json()
            location = LocationRequest(**data)
            
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
                await websocket.send_json({"error": "Address geocoding not implemented"})
                continue
            
            async for result in process_single_roof_streaming(
                bbox,
                debug=location.debug,
                fallback=location.fallback_to_lower_quality,
                websocket=websocket
            ):
                await websocket.send_json(result)
                
    except WebSocketDisconnect:
        manager.disconnect(websocket)
    except Exception as e:
        logger.error(f"WebSocket error: {str(e)}")
        await websocket.send_json({"error": str(e)})
        manager.disconnect(websocket)

# --- GPU Memory Management ---
def clear_gpu_memory():
    if config.USE_GPU:
        torch.cuda.empty_cache()
        gc.collect()

# --- API Endpoints with Streaming Support ---
@app.post("/measure", response_model=List[MeasurementResult])
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

        # Process all buildings in the area
        results = []
        async for result in process_single_roof_streaming(
            bbox,
            debug=location.debug,
            fallback=location.fallback_to_lower_quality
        ):
            if "error" in result:
                raise HTTPException(500, result["error"])
            results.append(result)
            
        # Schedule memory cleanup
        background_tasks.add_task(clear_gpu_memory)
        return results
        
    except Exception as e:
        logger.error(f"Error processing request: {str(e)}")
        raise HTTPException(500, str(e))

# --- Colab Specific Setup ---
def setup_colab():
    """Initialize environment for Google Colab"""
    if 'COLAB_GPU' in os.environ:
        logger.info("Running in Google Colab environment")
        # Mount Google Drive if needed
        if not os.path.exists('/content/drive'):
            from google.colab import drive
            drive.mount('/content/drive')
        
        # Install required system dependencies
        os.system('apt-get install -y gdal-bin libgdal-dev')
        os.system('pip install rasterio --upgrade')
        
        # Verify GPU availability
        if torch.cuda.is_available():
            logger.info(f"Using GPU: {torch.cuda.get_device_name(0)}")
        else:
            logger.warning("No GPU available in Colab")

# --- Run ---
if __name__ == "__main__":
    # Setup Colab environment if detected
    if 'COLAB_GPU' in os.environ:
        setup_colab()
    
    # Start the server
    import uvicorn
    uvicorn.run(
        app, 
        host="0.0.0.0", 
        port=8000,
        workers=1 if config.USE_GPU else 4,  # Fewer workers for GPU to avoid OOM
        timeout_keep_alive=30
    )
