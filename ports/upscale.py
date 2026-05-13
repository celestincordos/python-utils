from abc import abstractmethod
from dataclasses import dataclass

from PIL.Image import Image


@dataclass
class UpscalePort:
    @abstractmethod
    async def upscale_image(
        self,
        image: Image,
        upscale_factor: int = 2,
        model_name: str = "imagegeneration@002",
    ) -> Image:
        """Upscales an image using a specified model and factor."""
        pass
