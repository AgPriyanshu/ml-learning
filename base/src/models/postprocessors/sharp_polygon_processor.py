"""
Enhanced polygon processor that handles blurred boundaries from tile blending.
Includes sharpening techniques and adaptive thresholding for better polygon extraction.
"""

import numpy as np
import rasterio
from pathlib import Path
from typing import Union, Tuple, Optional, List, Dict, Any
from scipy import ndimage
from skimage import filters, morphology, measure
import cv2

from .polygon_processor import BuildingPolygonProcessor


class SharpPolygonProcessor(BuildingPolygonProcessor):
    """
    Enhanced polygon processor that addresses blurred boundaries from tile blending.
    Includes boundary sharpening and adaptive thresholding methods.
    """
    
    def __init__(
        self,
        min_area: float = 50.0,
        simplify_tolerance: float = 1.0,
        buffer_distance: float = 0.5,
        morphology_kernel_size: int = 3,
        use_morphology: bool = True,
        # New sharpening parameters
        sharpening_method: str = "adaptive_threshold",  # "adaptive_threshold", "unsharp_mask", "gradient"
        adaptive_block_size: int = 15,  # Block size for adaptive thresholding
        adaptive_c: float = 2.0,  # Constant subtracted from mean in adaptive thresholding
        unsharp_radius: float = 1.0,  # Radius for unsharp masking
        unsharp_amount: float = 1.5,  # Amount for unsharp masking
        edge_enhancement: bool = True,  # Whether to enhance edges before polygon extraction
    ):
        """
        Initialize the sharp polygon processor.
        
        Args:
            sharpening_method: Method for boundary sharpening
                - "adaptive_threshold": Use adaptive thresholding for sharp boundaries
                - "unsharp_mask": Apply unsharp masking to enhance edges
                - "gradient": Use gradient-based edge enhancement
                - "none": No sharpening (use original processor)
            adaptive_block_size: Block size for adaptive thresholding (must be odd)
            adaptive_c: Constant for adaptive thresholding
            unsharp_radius: Radius for unsharp mask filter
            unsharp_amount: Strength of unsharp mask enhancement
            edge_enhancement: Whether to apply additional edge enhancement
        """
        super().__init__(
            min_area=min_area,
            simplify_tolerance=simplify_tolerance,
            buffer_distance=buffer_distance,
            morphology_kernel_size=morphology_kernel_size,
            use_morphology=use_morphology,
        )
        
        self.sharpening_method = sharpening_method
        self.adaptive_block_size = adaptive_block_size if adaptive_block_size % 2 == 1 else adaptive_block_size + 1
        self.adaptive_c = adaptive_c
        self.unsharp_radius = unsharp_radius
        self.unsharp_amount = unsharp_amount
        self.edge_enhancement = edge_enhancement

    def apply_unsharp_mask(self, prob_map: np.ndarray) -> np.ndarray:
        """
        Apply unsharp masking to enhance edges in probability map.
        
        Args:
            prob_map: Probability map (0-1 range)
            
        Returns:
            Enhanced probability map
        """
        # Convert to 8-bit for OpenCV
        prob_8bit = (prob_map * 255).astype(np.uint8)
        
        # Create Gaussian blur
        blurred = cv2.GaussianBlur(prob_8bit, (0, 0), self.unsharp_radius)
        
        # Create unsharp mask
        unsharp = cv2.addWeighted(prob_8bit, 1 + self.unsharp_amount, blurred, -self.unsharp_amount, 0)
        
        # Convert back to 0-1 range
        return np.clip(unsharp.astype(np.float32) / 255.0, 0, 1)

    def apply_gradient_enhancement(self, prob_map: np.ndarray) -> np.ndarray:
        """
        Apply gradient-based edge enhancement.
        
        Args:
            prob_map: Probability map (0-1 range)
            
        Returns:
            Enhanced probability map
        """
        # Calculate gradients
        grad_x = np.abs(cv2.Sobel(prob_map, cv2.CV_64F, 1, 0, ksize=3))
        grad_y = np.abs(cv2.Sobel(prob_map, cv2.CV_64F, 0, 1, ksize=3))
        
        # Combine gradients
        gradient_magnitude = np.sqrt(grad_x**2 + grad_y**2)
        
        # Normalize gradient
        gradient_magnitude = gradient_magnitude / gradient_magnitude.max() if gradient_magnitude.max() > 0 else gradient_magnitude
        
        # Enhance edges by adding gradient information
        enhanced = prob_map + (gradient_magnitude * 0.3)
        
        return np.clip(enhanced, 0, 1)

    def apply_adaptive_threshold(self, prob_map: np.ndarray, base_threshold: float = 0.5) -> np.ndarray:
        """
        Apply adaptive thresholding to create sharper boundaries.
        
        Args:
            prob_map: Probability map (0-1 range)
            base_threshold: Base threshold value
            
        Returns:
            Binary mask with sharp boundaries
        """
        # Convert to 8-bit for adaptive thresholding
        prob_8bit = (prob_map * 255).astype(np.uint8)
        
        # Apply adaptive threshold
        adaptive_mask = cv2.adaptiveThreshold(
            prob_8bit,
            255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY,
            self.adaptive_block_size,
            self.adaptive_c
        )
        
        # Combine with global threshold
        global_mask = (prob_map >= base_threshold).astype(np.uint8) * 255
        
        # Use intersection of both methods for more conservative detection
        combined_mask = cv2.bitwise_and(adaptive_mask, global_mask)
        
        return (combined_mask > 0).astype(np.uint8)

    def enhance_probability_map(self, prob_map: np.ndarray) -> np.ndarray:
        """
        Enhance probability map to create sharper boundaries.
        
        Args:
            prob_map: Input probability map (0-1 range)
            
        Returns:
            Enhanced probability map
        """
        enhanced = prob_map.copy()
        
        if self.sharpening_method == "unsharp_mask":
            enhanced = self.apply_unsharp_mask(enhanced)
        elif self.sharpening_method == "gradient":
            enhanced = self.apply_gradient_enhancement(enhanced)
        
        # Optional additional edge enhancement
        if self.edge_enhancement and self.sharpening_method not in ["adaptive_threshold"]:
            # Slight edge enhancement using Laplacian
            laplacian = cv2.Laplacian(enhanced, cv2.CV_64F)
            enhanced = enhanced + (laplacian * 0.1)
            enhanced = np.clip(enhanced, 0, 1)
        
        return enhanced

    def create_sharp_mask(self, prob_map: np.ndarray, threshold: float = 0.5) -> np.ndarray:
        """
        Create a sharp binary mask from probability map.
        
        Args:
            prob_map: Probability map (0-1 range)
            threshold: Probability threshold
            
        Returns:
            Sharp binary mask
        """
        if self.sharpening_method == "adaptive_threshold":
            return self.apply_adaptive_threshold(prob_map, threshold)
        else:
            # Enhance probability map first, then threshold
            enhanced = self.enhance_probability_map(prob_map)
            return (enhanced >= threshold).astype(np.uint8)

    def process_probability_raster_sharp(
        self,
        prob_path: Union[str, Path],
        threshold: float,
        output_dir: Union[str, Path],
        base_name: Optional[str] = None,
        save_enhanced: bool = False
    ) -> Tuple[str, str, int]:
        """
        Process probability raster with boundary sharpening.
        
        Args:
            prob_path: Path to probability raster
            threshold: Probability threshold for building detection
            output_dir: Output directory for shapefiles
            base_name: Base name for output files
            save_enhanced: Whether to save the enhanced probability map
            
        Returns:
            Tuple of (buildings_shapefile_path, bboxes_shapefile_path, num_buildings)
        """
        prob_path = Path(prob_path)
        output_dir = Path(output_dir)
        if base_name is None:
            base_name = prob_path.stem

        # Load probability raster
        with rasterio.open(prob_path) as src:
            prob = src.read(1).astype(np.float32)
            transform = src.transform
            crs = src.crs.to_string() if src.crs else "EPSG:4326"
            profile = src.profile.copy()

        # Normalize probability to 0-1 if needed
        if prob.max() > 1:
            prob = prob / 255.0

        print(f"Applying {self.sharpening_method} boundary sharpening...")
        
        # Create sharp binary mask
        sharp_mask = self.create_sharp_mask(prob, threshold)
        
        # Save enhanced probability map if requested
        if save_enhanced:
            enhanced_prob = self.enhance_probability_map(prob)
            enhanced_path = output_dir / f"{base_name}_enhanced_prob.tif"
            
            profile.update(dtype='float32', count=1)
            with rasterio.open(enhanced_path, 'w', **profile) as dst:
                dst.write(enhanced_prob, 1)
            print(f"Saved enhanced probability map: {enhanced_path}")

        # Extract polygons from sharp mask
        polygons = self.extract_polygons(sharp_mask, transform, crs)
        print(f"Extracted {len(polygons)} building polygons using sharp boundaries")

        # Create GeoDataFrames
        buildings_gdf, bboxes_gdf = self.create_geodataframes(polygons, crs)

        # Save shapefiles
        buildings_path, bboxes_path = self.save_shapefiles(
            buildings_gdf, bboxes_gdf, output_dir, f"{base_name}_sharp"
        )

        return buildings_path, bboxes_path, len(polygons)


# Convenience function for sharp polygon extraction
def extract_sharp_building_vectors(
    prob_path: Union[str, Path],
    threshold: float,
    output_dir: Union[str, Path],
    min_area: float = 50.0,
    sharpening_method: str = "adaptive_threshold",
    base_name: Optional[str] = None
) -> Tuple[str, str, int]:
    """
    Convenience function for extracting sharp building vectors from probability raster.
    
    Args:
        prob_path: Path to probability raster
        threshold: Probability threshold
        output_dir: Output directory for shapefiles
        min_area: Minimum building area in square meters
        sharpening_method: Boundary sharpening method
        base_name: Base name for output files
        
    Returns:
        Tuple of (buildings_shapefile_path, bboxes_shapefile_path, num_buildings)
    """
    processor = SharpPolygonProcessor(
        min_area=min_area,
        sharpening_method=sharpening_method
    )
    return processor.process_probability_raster_sharp(
        prob_path, threshold, output_dir, base_name, save_enhanced=True
    )
