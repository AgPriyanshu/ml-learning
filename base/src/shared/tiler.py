"""
Enhanced PyVips-based Tiler with optimized batching and memory management.
"""
import numpy as np
import pyvips
from pathlib import Path
from typing import Iterator, Tuple, Optional, List, Union, Dict, Any
import logging
from dataclasses import dataclass
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from queue import Queue, Empty
import gc
import psutil

logger = logging.getLogger(__name__)

@dataclass
class TileInfo:
    """Information about a tile."""
    x: int
    y: int
    width: int
    height: int
    actual_width: int
    actual_height: int
    tile_id: int

@dataclass
class BatchConfig:
    """Configuration for optimized batching."""
    batch_size: int = 8
    prefetch_batches: int = 2  # Number of batches to prefetch
    max_workers: int = 4       # Number of worker threads
    memory_limit_mb: int = 2048  # Memory limit for caching
    enable_cache: bool = True   # Enable tile caching
    cache_size: int = 100      # Number of tiles to cache

class TileCache:
    """LRU cache for tiles with memory management."""
    
    def __init__(self, max_size_mb: int = 1024, max_items: int = 100):
        self.max_size_mb = max_size_mb
        self.max_items = max_items
        self._cache: Dict[int, np.ndarray] = {}
        self._access_order: List[int] = []
        self._current_size_mb = 0
        self._lock = threading.Lock()
        
    def get(self, tile_id: int) -> Optional[np.ndarray]:
        """Get tile from cache."""
        with self._lock:
            if tile_id in self._cache:
                # Update access order
                self._access_order.remove(tile_id)
                self._access_order.append(tile_id)
                return self._cache[tile_id].copy()
        return None
    
    def put(self, tile_id: int, tile_data: np.ndarray):
        """Put tile in cache."""
        tile_size_mb = tile_data.nbytes / (1024 * 1024)
        
        with self._lock:
            # Remove if already exists
            if tile_id in self._cache:
                self._current_size_mb -= self._cache[tile_id].nbytes / (1024 * 1024)
                del self._cache[tile_id]
                self._access_order.remove(tile_id)
            
            # Check if we need to evict
            while (len(self._cache) >= self.max_items or 
                   self._current_size_mb + tile_size_mb > self.max_size_mb):
                if not self._access_order:
                    break
                
                oldest_id = self._access_order.pop(0)
                if oldest_id in self._cache:
                    self._current_size_mb -= self._cache[oldest_id].nbytes / (1024 * 1024)
                    del self._cache[oldest_id]
            
            # Add new tile
            self._cache[tile_id] = tile_data.copy()
            self._access_order.append(tile_id)
            self._current_size_mb += tile_size_mb
    
    def clear(self):
        """Clear the cache."""
        with self._lock:
            self._cache.clear()
            self._access_order.clear()
            self._current_size_mb = 0
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        with self._lock:
            return {
                'size_mb': self._current_size_mb,
                'num_items': len(self._cache),
                'hit_ratio': getattr(self, '_hits', 0) / max(getattr(self, '_requests', 1), 1)
            }

class Tiler:
    """
    Enhanced memory-efficient tiler with optimized batching.
    
    Features:
    - Parallel tile extraction
    - Pre-fetching with background workers
    - Memory pooling for array reuse
    - Configurable caching
    - Memory usage monitoring
    - GPU memory optimization
    """
    
    def __init__(
        self,
        image_path: Union[str, Path],
        tile_size: int = 512,
        overlap: int = 0,
        page: int = 0,
        min_tile_coverage: float = 0.5,
        batch_config: Optional[BatchConfig] = None,
        validate_on_init: bool = True
    ):
        """
        Initialize the OptimizedTiler.
        
        Args:
            image_path: Path to the TIFF file
            tile_size: Size of square tiles to extract
            overlap: Overlap between adjacent tiles in pixels
            page: Page/level to read from (for multi-page/pyramidal TIFFs)
            min_tile_coverage: Minimum coverage ratio for edge tiles (0.0-1.0)
            batch_config: Configuration for batching optimizations
            validate_on_init: Validate the image file on initialization
        """
        self.image_path = Path(image_path)
        self.tile_size = tile_size
        self.overlap = overlap
        self.page = page
        self.min_tile_coverage = min_tile_coverage
        
        # Batch configuration
        self.batch_config = batch_config or BatchConfig()
        
        # Validate input parameters
        if not self.image_path.exists():
            raise FileNotFoundError(f"Image file not found: {self.image_path}")
        
        if tile_size <= 0:
            raise ValueError("tile_size must be positive")
            
        if overlap < 0:
            raise ValueError("overlap must be non-negative")
            
        if overlap >= tile_size:
            raise ValueError("overlap must be less than tile_size")
            
        if not 0 <= min_tile_coverage <= 1:
            raise ValueError("min_tile_coverage must be between 0 and 1")
        
        # Load image with PyVips (memory efficient)
        try:
            if page > 0:
                self.image = pyvips.Image.new_from_file(str(self.image_path), page=page)
            else:
                self.image = pyvips.Image.new_from_file(str(self.image_path))
        except Exception as e:
            raise RuntimeError(f"Failed to open image with PyVips: {e}")
        
        # Image properties
        self.width = self.image.width
        self.height = self.image.height
        self.bands = self.image.bands
        self.format = self.image.format
        
        # Calculate tile grid
        self.step_size = tile_size - overlap
        self.tiles_x = self._calculate_tiles_along_axis(self.width)
        self.tiles_y = self._calculate_tiles_along_axis(self.height)
        self.total_tiles = self.tiles_x * self.tiles_y
        
        # Generate tile coordinates
        self.tile_coords = self._generate_tile_coordinates()
        
        # Initialize optimization components
        self.tile_cache = TileCache(
            max_size_mb=self.batch_config.memory_limit_mb // 2,
            max_items=self.batch_config.cache_size
        ) if self.batch_config.enable_cache else None
        
        # Threading components
        self._executor = None
        self._prefetch_queue = Queue(maxsize=self.batch_config.prefetch_batches * 2)
        self._stop_prefetch = threading.Event()
        
        if validate_on_init:
            self._validate_tiling()
        
        logger.info(f"Initialized OptimizedTiler for {self.image_path.name}")
        logger.info(f"Image: {self.width}x{self.height}x{self.bands}, Format: {self.format}")
        logger.info(f"Tiles: {self.tiles_x}x{self.tiles_y} = {self.total_tiles} tiles")
        logger.info(f"Batch config: {self.batch_config}")
    
    
    def __len__(self) -> int:
        """Return number of tiles."""
        return len(self.tile_coords)

    def __getitem__(self, idx: int) -> TileInfo:
        """Get tile by index."""
        return self.tile_coords[idx]

    
    def __del__(self):
        """Cleanup resources."""
        self._stop_prefetch.set()
        if self._executor:
            self._executor.shutdown(wait=False)
        self.clear_cache()
    
    def __repr__(self) -> str:
        return (f"OptimizedTiler(path='{self.image_path.name}', "
                f"size={self.width}x{self.height}, "
                f"tiles={self.total_tiles}, "
                f"tile_size={self.tile_size}, "
                f"batch_size={self.batch_config.batch_size})")

    def get_tile(self, tile_info: TileInfo, as_numpy: bool = True) -> Union[np.ndarray, pyvips.Image]:
        """Extract a single tile with optimizations."""
        return self._extract_tile_optimized(tile_info, as_numpy)
    
    def get_tile_by_id(self, tile_id: int, as_numpy: bool = True) -> Union[np.ndarray, pyvips.Image]:
        """Get tile by its ID."""
        if not 0 <= tile_id < len(self.tile_coords):
            raise IndexError(f"Tile ID {tile_id} out of range [0, {len(self.tile_coords)})")
        
        return self.get_tile(self.tile_coords[tile_id], as_numpy=as_numpy)
    
    def get_memory_usage_estimate(self) -> Dict[str, Any]:
        """Get detailed memory usage estimates."""
        single_tile_mb = (self.tile_size * self.tile_size * self.bands * 4) / (1024 * 1024)
        full_image_gb = (self.width * self.height * self.bands * 4) / (1024**3)
        batch_mb = single_tile_mb * self.batch_config.batch_size
        
        # Current system memory
        memory = psutil.virtual_memory()
        
        estimate = {
            'single_tile_mb': single_tile_mb,
            'batch_mb': batch_mb,
            'full_image_gb': full_image_gb,
            'total_tiles': self.total_tiles,
            'recommended_batch_size': max(1, int(self.batch_config.memory_limit_mb / single_tile_mb)),
            'system_memory_gb': memory.total / (1024**3),
            'available_memory_gb': memory.available / (1024**3),
            'memory_usage_percent': memory.percent
        }
        
        if self.tile_cache:
            estimate['cache_stats'] = self.tile_cache.get_stats()
        
        return estimate
    
    def iter_tiles_optimized(
        self, 
        batch_size: Optional[int] = None,
        as_numpy: bool = True,
        parallel: bool = True,
        prefetch: bool = True
    ) -> Iterator[List[Tuple[TileInfo, Union[np.ndarray, pyvips.Image]]]]:
        """
        Iterate over tiles in optimized batches.
        
        Args:
            batch_size: Size of batches (uses config default if None)
            as_numpy: Return tiles as numpy arrays
            parallel: Use parallel tile extraction
            prefetch: Enable prefetching of next batch
            
        Yields:
            List of (tile_info, tile_data) tuples
        """
        batch_size = batch_size or self.batch_config.batch_size
        
        # Split tiles into batches
        tile_batches = [
            self.tile_coords[i:i + batch_size]
            for i in range(0, len(self.tile_coords), batch_size)
        ]
        
        if not prefetch or len(tile_batches) <= 1:
            # Simple mode - no prefetching
            for tile_batch in tile_batches:
                if parallel and len(tile_batch) > 1:
                    batch_results = self._extract_tiles_parallel(tile_batch, as_numpy)
                else:
                    batch_results = [
                        (tile_info, self._extract_tile_optimized(tile_info, as_numpy))
                        for tile_info in tile_batch
                    ]
                yield batch_results
        else:
            # Prefetching mode
            yield from self._iter_tiles_with_prefetch(tile_batches, as_numpy, parallel)
    
    def clear_cache(self):
        """Clear all caches and free memory."""
        if self.tile_cache:
            self.tile_cache.clear()
        
        gc.collect()

    def _calculate_tiles_along_axis(self, axis_size: int) -> int:
        """Calculate number of tiles along an axis."""
        if axis_size <= self.tile_size:
            return 1
        return int(np.ceil((axis_size - self.tile_size) / self.step_size)) + 1
    
    def _generate_tile_coordinates(self) -> List[TileInfo]:
        """Generate coordinates for all tiles."""
        coords = []
        tile_id = 0
        
        for j in range(self.tiles_y):
            for i in range(self.tiles_x):
                x = i * self.step_size
                y = j * self.step_size
                
                actual_width = min(self.tile_size, self.width - x)
                actual_height = min(self.tile_size, self.height - y)
                
                coverage = (actual_width * actual_height) / (self.tile_size * self.tile_size)
                
                if coverage >= self.min_tile_coverage:
                    coords.append(TileInfo(
                        x=x, y=y,
                        width=self.tile_size, height=self.tile_size,
                        actual_width=actual_width, actual_height=actual_height,
                        tile_id=tile_id
                    ))
                    tile_id += 1
        
        return coords
    
    def _validate_tiling(self):
        """Validate the tiling configuration."""
        if not self.tile_coords:
            raise ValueError("No valid tiles generated. Check min_tile_coverage setting.")
        
        estimated_gb = (self.width * self.height * self.bands * 4) / (1024**3)
        if estimated_gb > 100:
            logger.warning(f"Large image detected ({estimated_gb:.1f}GB). "
                         "Using optimized streaming processing.")
    
    def _standardize_array_format(self, array: np.ndarray) -> np.ndarray:
      """
        Standardize array format to CHW (Channels, Height, Width) for PyTorch.
        By default, PyVips/PIL returns the array in HWC (Height, Width, Channels) format.
        
        Args:
            array: Input array in various formats
            
        Returns:
            Array in CHW format
      """
      if array.ndim == 2:
          # Grayscale (H, W) → (1, H, W)
          return array[np.newaxis, ...]
      elif array.ndim == 3 and array.shape[2] <= 4:
          # Multi-channel (H, W, C) → (C, H, W)
          return array.transpose(2, 0, 1)
      else:
          # Return as-is for other formats
          return array

    def _extract_tile_optimized(self, tile_info: TileInfo, as_numpy: bool = True) -> Union[np.ndarray, pyvips.Image]:
        """Extract a single tile with optimizations."""
        # Check cache first
        if self.tile_cache and as_numpy:
            cached_tile = self.tile_cache.get(tile_info.tile_id)
            if cached_tile is not None:
                return cached_tile
        
        try:
            # Extract tile using PyVips
            tile: pyvips.Image = self.image.crop(
                tile_info.x, tile_info.y, 
                tile_info.actual_width, tile_info.actual_height
            )
            
            if as_numpy:
                # Create new array
                tile_array = self._standardize_array_format(tile.numpy())
                
                # Pad if needed
                if (tile_info.actual_width < tile_info.width or 
                    tile_info.actual_height < tile_info.height):
                    tile_array = self._pad_tile(tile_array, tile_info)
                
                # Cache the result
                if self.tile_cache:
                    self.tile_cache.put(tile_info.tile_id, tile_array)
                
                return tile_array
            else:
                return tile
                
        except Exception as e:
            logger.error(f"Failed to extract tile {tile_info.tile_id}: {e}")
            raise
    
    def _pad_tile(self, tile_array: np.ndarray, tile_info: TileInfo) -> np.ndarray:
        """Pad tile to requested size."""
        c, h, w = tile_array.shape
        target_h, target_w = tile_info.height, tile_info.width
        
        if h == target_h and w == target_w:
            return tile_array
        
        pad_h = target_h - h
        pad_w = target_w - w
        
        # Use reflection padding
        padded = np.pad(
            tile_array,
            ((0, 0), (0, pad_h), (0, pad_w)),
            mode='reflect'
        )
        
        return padded
    
    def _extract_tiles_parallel(self, tile_infos: List[TileInfo], as_numpy: bool = True) -> List[Tuple[TileInfo, Union[np.ndarray, pyvips.Image]]]:
        """Extract multiple tiles in parallel."""
        if len(tile_infos) == 1:
            # Single tile - no need for threading overhead
            tile_info = tile_infos[0]
            tile_data = self._extract_tile_optimized(tile_info, as_numpy)
            return [(tile_info, tile_data)]
        
        results = []
        with ThreadPoolExecutor(max_workers=self.batch_config.max_workers) as executor:
            # Submit all tile extraction tasks
            future_to_tile = {
                executor.submit(self._extract_tile_optimized, tile_info, as_numpy): tile_info
                for tile_info in tile_infos
            }
            
            # Collect results as they complete
            for future in as_completed(future_to_tile):
                tile_info = future_to_tile[future]
                try:
                    tile_data = future.result()
                    results.append((tile_info, tile_data))
                except Exception as e:
                    logger.error(f"Failed to extract tile {tile_info.tile_id}: {e}")
                    continue
        
        # Sort results by tile_id to maintain order
        results.sort(key=lambda x: x[0].tile_id)
        return results

    def _iter_tiles_with_prefetch(
        self, 
        tile_batches: List[List[TileInfo]], 
        as_numpy: bool, 
        parallel: bool
    ) -> Iterator[List[Tuple[TileInfo, Union[np.ndarray, pyvips.Image]]]]:
        """Iterate with prefetching using background workers."""
        if not tile_batches:
            return
        
        self._stop_prefetch.clear()
        
        def prefetch_worker():
            """Background worker for prefetching batches."""
            for i, tile_batch in enumerate(tile_batches):
                if self._stop_prefetch.is_set():
                    break
                
                try:
                    if parallel and len(tile_batch) > 1:
                        batch_results = self._extract_tiles_parallel(tile_batch, as_numpy)
                    else:
                        batch_results = [
                            (tile_info, self._extract_tile_optimized(tile_info, as_numpy))
                            for tile_info in tile_batch
                        ]
                    
                    self._prefetch_queue.put((i, batch_results), timeout=30)
                except Exception as e:
                    logger.error(f"Prefetch error for batch {i}: {e}")
                    self._prefetch_queue.put((i, None))
        
        # Start prefetch worker
        prefetch_thread = threading.Thread(target=prefetch_worker, daemon=True)
        prefetch_thread.start()
        
        try:
            # Yield batches in order
            for expected_batch_id in range(len(tile_batches)):
                try:
                    batch_id, batch_results = self._prefetch_queue.get(timeout=60)
                    if batch_results is not None:
                        yield batch_results
                    else:
                        logger.warning(f"Skipping failed batch {batch_id}")
                except Empty:
                    logger.error(f"Timeout waiting for batch {expected_batch_id}")
                    break
        finally:
            self._stop_prefetch.set()
            prefetch_thread.join(timeout=5)
            
            # Clear any remaining items in queue
            while not self._prefetch_queue.empty():
                try:
                    self._prefetch_queue.get_nowait()
                except Empty:
                    break
 