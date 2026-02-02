"""Engine registry for managing multiple style transfer engines."""

import os
from pathlib import Path
from typing import Dict, Type

from engines.base_engine import BaseStyleEngine
from engines.adain_engine import AdaINEngine
from engines.ghibli_diffusion_engine import GhibliDiffusionEngine


def _default_engine_from_env() -> str:
    """Read default engine from DEFAULT_ENGINE env var. Fallback: adain."""
    name = (os.environ.get("DEFAULT_ENGINE") or "").strip().lower()
    allowed = {"adain", "ghibli_diffusion"}
    return name if name in allowed else "adain"


class EngineRegistry:
    """Registry for managing multiple style transfer engines."""

    def __init__(self):
        self._engines: Dict[str, BaseStyleEngine] = {}
        self._engine_classes: Dict[str, Type[BaseStyleEngine]] = {
            "adain": AdaINEngine,
            "ghibli_diffusion": GhibliDiffusionEngine,
        }
        self._default_engine_name = _default_engine_from_env()

    def register_engine_class(self, name: str, engine_class: Type[BaseStyleEngine]) -> None:
        """Register a new engine class."""
        self._engine_classes[name] = engine_class

    def get_engine(self, name: str, **kwargs) -> BaseStyleEngine:
        """Get or create an engine instance."""
        if name not in self._engines:
            if name not in self._engine_classes:
                raise ValueError(f"Unknown engine: {name}. Available: {list(self._engine_classes.keys())}")
            
            engine_class = self._engine_classes[name]
            self._engines[name] = engine_class(**kwargs)
            self._engines[name].load_models()
        
        return self._engines[name]

    def list_engines(self) -> Dict[str, dict]:
        """List all available engines with metadata."""
        result = {}
        for name, engine_class in self._engine_classes.items():
            # Create temporary instance to get metadata
            temp_engine = engine_class()
            result[name] = {
                "name": temp_engine.name,
                "speed": temp_engine.speed_rating,
                "quality": temp_engine.quality_rating,
                "loaded": name in self._engines,
            }
        return result

    def set_default_engine(self, name: str) -> None:
        """Set default engine."""
        if name not in self._engine_classes:
            raise ValueError(f"Unknown engine: {name}")
        self._default_engine_name = name

    def get_default_engine(self, **kwargs) -> BaseStyleEngine:
        """Get default engine instance."""
        return self.get_engine(self._default_engine_name, **kwargs)

    @property
    def default_engine_name(self) -> str:
        """Get default engine name."""
        return self._default_engine_name


# Global registry instance
registry = EngineRegistry()
