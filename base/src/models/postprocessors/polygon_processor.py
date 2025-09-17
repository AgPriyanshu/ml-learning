"""
Post-processing module for converting building segmentation masks to polygon vectors and shapefiles.
"""

import os
import tempfile
from pathlib import Path
from typing import List, Tuple, Optional, Union, Dict, Any

import numpy as np
import rasterio
from rasterio import features
from rasterio.transform import from_bounds
import geopandas as gpd
from shapely.geometry import Polygon, Point, box
from shapely.ops import unary_union
from shapely.validation import make_valid
import cv2
from scipy import ndimage
from skimage import measure, morphology


class BuildingPolygonProcessor:
    """
    Post-processor for extracting building polygons from segmentation masks
    and generating shapefiles with bounding box information.
    """

    def __init__(
        self,
        min_area: float = 50.0,  # Minimum building area in square meters
        simplify_tolerance: float = 1.0,  # Polygon simplification tolerance
        buffer_distance: float = 0.5,  # Buffer distance for polygon cleaning
        morphology_kernel_size: int = 3,  # Kernel size for morphological operations
        use_morphology: bool = True,  # Whether to apply morphological operations
    ):
        """
        Initialize the polygon processor.
        
        Args:
            min_area: Minimum area threshold for building polygons (square meters)
            simplify_tolerance: Tolerance for polygon simplification (meters)
            buffer_distance: Buffer distance for polygon cleaning (meters)
            morphology_kernel_size: Kernel size for morphological operations
            use_morphology: Whether to apply morphological operations for noise reduction
        """
        self.min_area = min_area
        self.simplify_tolerance = simplify_tolerance
        self.buffer_distance = buffer_distance
        self.morphology_kernel_size = morphology_kernel_size
        self.use_morphology = use_morphology

    def clean_mask(self, mask: np.ndarray) -> np.ndarray:
        """
        Clean the binary mask using morphological operations.
        
        Args:
            mask: Binary mask (0/1 or 0/255)
            
        Returns:
            Cleaned binary mask
        """
        # Ensure binary mask
        if mask.max() > 1:
            mask = (mask > 127).astype(np.uint8)
        else:
            mask = mask.astype(np.uint8)

        if not self.use_morphology:
            return mask

        # Define morphological kernel
        kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, 
            (self.morphology_kernel_size, self.morphology_kernel_size)
        )

        # Remove small noise
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
        
        # Fill small holes
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=1)
        
        # Additional hole filling for larger holes
        mask = ndimage.binary_fill_holes(mask).astype(np.uint8)

        return mask

    def extract_polygons(
        self, 
        mask: np.ndarray, 
        transform: rasterio.transform.Affine,
        crs: str = "EPSG:4326"
    ) -> List[Dict[str, Any]]:
        """
        Extract building polygons from a binary mask.
        
        Args:
            mask: Binary segmentation mask
            transform: Rasterio affine transform for georeferencing
            crs: Coordinate reference system
            
        Returns:
            List of polygon dictionaries with geometry and properties
        """
        # Clean the mask
        cleaned_mask = self.clean_mask(mask)

        # Extract polygon features using rasterio
        polygons = []
        
        # Generate polygon features from mask
        for geom, value in features.shapes(
            cleaned_mask.astype(np.uint8), 
            mask=cleaned_mask > 0,
            transform=transform
        ):
            if value == 1:  # Building pixels
                try:
                    # Create Shapely polygon
                    poly = Polygon(geom['coordinates'][0])
                    
                    # Validate and fix geometry if needed
                    if not poly.is_valid:
                        poly = make_valid(poly)
                    
                    # Skip if still invalid or empty
                    if not poly.is_valid or poly.is_empty:
                        continue
                    
                    # Calculate area in square meters (assuming CRS is in meters, adjust if needed)
                    if crs == "EPSG:4326":
                        # For geographic coordinates, approximate area calculation
                        # Convert to approximate meters using lat/lon bounds
                        bounds = poly.bounds
                        lat_center = (bounds[1] + bounds[3]) / 2
                        # Rough conversion: 1 degree lat ≈ 111,320 m, 1 degree lon ≈ 111,320 * cos(lat)
                        lat_m_per_deg = 111320
                        lon_m_per_deg = 111320 * np.cos(np.radians(lat_center))
                        
                        # Create a temporary polygon in meters for area calculation
                        coords = np.array(poly.exterior.coords)
                        coords_m = coords.copy()
                        coords_m[:, 0] *= lon_m_per_deg
                        coords_m[:, 1] *= lat_m_per_deg
                        poly_m = Polygon(coords_m)
                        area = poly_m.area
                    else:
                        area = poly.area
                    
                    # Filter by minimum area
                    if area < self.min_area:
                        continue
                    
                    # Simplify polygon
                    if self.simplify_tolerance > 0:
                        poly = poly.simplify(self.simplify_tolerance, preserve_topology=True)
                    
                    # Apply buffer for cleaning (and then negative buffer to restore size)
                    if self.buffer_distance > 0:
                        poly = poly.buffer(self.buffer_distance).buffer(-self.buffer_distance)
                    
                    # Skip if polygon became invalid after processing
                    if not poly.is_valid or poly.is_empty:
                        continue
                    
                    # Calculate bounding box
                    bounds = poly.bounds
                    bbox = box(bounds[0], bounds[1], bounds[2], bounds[3])
                    
                    # Calculate polygon properties
                    centroid = poly.centroid
                    
                    polygon_data = {
                        'geometry': poly,
                        'bbox_geometry': bbox,
                        'properties': {
                            'area_sqm': area,
                            'perimeter_m': poly.length,
                            'centroid_x': centroid.x,
                            'centroid_y': centroid.y,
                            'bbox_area_sqm': bbox.area if crs != "EPSG:4326" else (
                                (bounds[2] - bounds[0]) * lon_m_per_deg * 
                                (bounds[3] - bounds[1]) * lat_m_per_deg
                            ),
                            'compactness': (4 * np.pi * area) / (poly.length ** 2) if poly.length > 0 else 0,
                            'bbox_width': bounds[2] - bounds[0],
                            'bbox_height': bounds[3] - bounds[1],
                        }
                    }
                    
                    polygons.append(polygon_data)
                    
                except Exception as e:
                    print(f"Warning: Failed to process polygon: {e}")
                    continue

        return polygons

    def create_geodataframes(
        self, 
        polygons: List[Dict[str, Any]], 
        crs: str = "EPSG:4326"
    ) -> Tuple[gpd.GeoDataFrame, gpd.GeoDataFrame]:
        """
        Create GeoDataFrames for building polygons and their bounding boxes.
        
        Args:
            polygons: List of polygon dictionaries
            crs: Coordinate reference system
            
        Returns:
            Tuple of (building_polygons_gdf, bounding_boxes_gdf)
        """
        if not polygons:
            # Return empty GeoDataFrames with proper schema
            empty_poly_gdf = gpd.GeoDataFrame(
                columns=['area_sqm', 'perimeter_m', 'centroid_x', 'centroid_y', 
                        'compactness', 'geometry'],
                crs=crs
            )
            empty_bbox_gdf = gpd.GeoDataFrame(
                columns=['building_id', 'bbox_area_sqm', 'bbox_width', 'bbox_height', 'geometry'],
                crs=crs
            )
            return empty_poly_gdf, empty_bbox_gdf

        # Create building polygons GeoDataFrame
        building_data = []
        bbox_data = []
        
        for i, poly_data in enumerate(polygons):
            building_id = f"building_{i+1:06d}"
            
            # Building polygon data
            building_props = poly_data['properties'].copy()
            building_props['building_id'] = building_id
            building_data.append({
                'geometry': poly_data['geometry'],
                **building_props
            })
            
            # Bounding box data
            bbox_data.append({
                'geometry': poly_data['bbox_geometry'],
                'building_id': building_id,
                'bbox_area_sqm': poly_data['properties']['bbox_area_sqm'],
                'bbox_width': poly_data['properties']['bbox_width'],
                'bbox_height': poly_data['properties']['bbox_height'],
            })

        buildings_gdf = gpd.GeoDataFrame(building_data, crs=crs)
        bboxes_gdf = gpd.GeoDataFrame(bbox_data, crs=crs)

        return buildings_gdf, bboxes_gdf

    def save_shapefiles(
        self,
        buildings_gdf: gpd.GeoDataFrame,
        bboxes_gdf: gpd.GeoDataFrame,
        output_dir: Union[str, Path],
        base_name: str = "buildings"
    ) -> Tuple[str, str]:
        """
        Save GeoDataFrames as shapefiles.
        
        Args:
            buildings_gdf: GeoDataFrame with building polygons
            bboxes_gdf: GeoDataFrame with bounding boxes
            output_dir: Output directory path
            base_name: Base name for output files
            
        Returns:
            Tuple of (buildings_shapefile_path, bboxes_shapefile_path)
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        buildings_path = output_dir / f"{base_name}_polygons.shp"
        bboxes_path = output_dir / f"{base_name}_bboxes.shp"

        # Save shapefiles
        if not buildings_gdf.empty:
            buildings_gdf.to_file(buildings_path, driver='ESRI Shapefile')
        else:
            print("Warning: No buildings found, creating empty shapefile")
            # Create empty shapefile with proper schema
            buildings_gdf.to_file(buildings_path, driver='ESRI Shapefile')

        if not bboxes_gdf.empty:
            bboxes_gdf.to_file(bboxes_path, driver='ESRI Shapefile')
        else:
            print("Warning: No bounding boxes found, creating empty shapefile")
            bboxes_gdf.to_file(bboxes_path, driver='ESRI Shapefile')

        return str(buildings_path), str(bboxes_path)

    def process_mask_to_shapefiles(
        self,
        mask_path: Union[str, Path],
        output_dir: Union[str, Path],
        base_name: Optional[str] = None
    ) -> Tuple[str, str, int]:
        """
        Complete pipeline: load mask, extract polygons, and save shapefiles.
        
        Args:
            mask_path: Path to binary mask raster
            output_dir: Output directory for shapefiles
            base_name: Base name for output files (defaults to mask filename)
            
        Returns:
            Tuple of (buildings_shapefile_path, bboxes_shapefile_path, num_buildings)
        """
        mask_path = Path(mask_path)
        if base_name is None:
            base_name = mask_path.stem

        # Load mask and geospatial information
        with rasterio.open(mask_path) as src:
            mask = src.read(1)
            transform = src.transform
            crs = src.crs.to_string() if src.crs else "EPSG:4326"

        # Extract polygons
        polygons = self.extract_polygons(mask, transform, crs)
        print(f"Extracted {len(polygons)} building polygons")

        # Create GeoDataFrames
        buildings_gdf, bboxes_gdf = self.create_geodataframes(polygons, crs)

        # Save shapefiles
        buildings_path, bboxes_path = self.save_shapefiles(
            buildings_gdf, bboxes_gdf, output_dir, base_name
        )

        return buildings_path, bboxes_path, len(polygons)

    def process_probability_raster(
        self,
        prob_path: Union[str, Path],
        threshold: float,
        output_dir: Union[str, Path],
        base_name: Optional[str] = None
    ) -> Tuple[str, str, int]:
        """
        Process probability raster: threshold to binary, extract polygons, save shapefiles.
        
        Args:
            prob_path: Path to probability raster
            threshold: Probability threshold for building detection
            output_dir: Output directory for shapefiles
            base_name: Base name for output files
            
        Returns:
            Tuple of (buildings_shapefile_path, bboxes_shapefile_path, num_buildings)
        """
        prob_path = Path(prob_path)
        if base_name is None:
            base_name = prob_path.stem

        # Load probability raster
        with rasterio.open(prob_path) as src:
            prob = src.read(1)
            transform = src.transform
            crs = src.crs.to_string() if src.crs else "EPSG:4326"

        # Apply threshold to create binary mask
        mask = (prob >= threshold).astype(np.uint8)

        # Extract polygons
        polygons = self.extract_polygons(mask, transform, crs)
        print(f"Extracted {len(polygons)} building polygons from probability raster")

        # Create GeoDataFrames
        buildings_gdf, bboxes_gdf = self.create_geodataframes(polygons, crs)

        # Save shapefiles
        buildings_path, bboxes_path = self.save_shapefiles(
            buildings_gdf, bboxes_gdf, output_dir, base_name
        )

        return buildings_path, bboxes_path, len(polygons)


# Convenience function for quick processing
def extract_building_vectors(
    mask_path: Union[str, Path],
    output_dir: Union[str, Path],
    min_area: float = 50.0,
    simplify_tolerance: float = 1.0,
    base_name: Optional[str] = None
) -> Tuple[str, str, int]:
    """
    Convenience function for extracting building vectors from a mask.
    
    Args:
        mask_path: Path to binary mask or probability raster
        output_dir: Output directory for shapefiles
        min_area: Minimum building area in square meters
        simplify_tolerance: Polygon simplification tolerance in meters
        base_name: Base name for output files
        
    Returns:
        Tuple of (buildings_shapefile_path, bboxes_shapefile_path, num_buildings)
    """
    processor = BuildingPolygonProcessor(
        min_area=min_area,
        simplify_tolerance=simplify_tolerance
    )
    return processor.process_mask_to_shapefiles(mask_path, output_dir, base_name)
