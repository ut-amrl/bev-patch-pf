from __future__ import annotations

import threading
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from geotiff.handler import GeoTiffHandler


class GeoTiffManager:
    """
    Singleton manager for GeoTiffHandler instances.
    Ensures that each geotiff file is loaded only once and shared across datasets.
    """

    _instance = None
    _lock = threading.Lock()

    def __new__(cls) -> GeoTiffManager:
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._handlers = {}
        return cls._instance

    def get_handler(self, geo_tiff_path: str | None = None) -> GeoTiffHandler | None:
        if geo_tiff_path is None:
            return None

        # Normalize path to ensure consistency
        path_key = str(Path(geo_tiff_path).resolve())

        if path_key not in self._handlers:
            # Lazy import to avoid circular dependencies
            from geotiff.handler import GeoTiffHandler

            self._handlers[path_key] = GeoTiffHandler(geo_tiff_path)

        return self._handlers[path_key]

    def clear_cache(self) -> None:
        self._handlers.clear()

    def get_cache_info(self) -> dict[str, int]:
        return {"num_handlers": len(self._handlers), "paths": list(self._handlers.keys())}


# Global instance for convenience
_manager = GeoTiffManager()


def get_geotiff_handler(geo_tiff_path: str | None = None) -> GeoTiffHandler | None:
    return _manager.get_handler(geo_tiff_path)
