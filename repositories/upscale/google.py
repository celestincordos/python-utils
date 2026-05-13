import io
from typing import Literal

from asgiref.sync import sync_to_async
from PIL.Image import Image
from vertexai.preview.vision_models import Image as VertexImage
from vertexai.preview.vision_models import ImageGenerationModel

from ports.upscale import UpscalePort


class GoogleUpscaleRepository(UpscalePort):
    async def upscale_image(
        self,
        image: Image,
        upscale_factor: int = 2,
        model_name: str = "imagegeneration@002",
    ) -> Image:
        if upscale_factor < 2:
            return image
        # Logic to upscale an image using Google services
        img_byte_io = io.BytesIO()

        image.save(img_byte_io, format="PNG")
        img_byte_arr = img_byte_io.getvalue()

        # Load image into Vertex AI format
        vertex_image = VertexImage(image_bytes=img_byte_arr)

        # Instantiate the model
        model = ImageGenerationModel.from_pretrained(model_name)

        # Call the upscaling method
        google_upscale_factor: Literal["x2", "x4"] = (
            "x2" if upscale_factor < 3 else "x4"
        )
        upscaled_vertex_img = await sync_to_async(model.upscale_image)(
            image=vertex_image,
            upscale_factor=google_upscale_factor,  # "x2" (default) or "x4"
        )

        # Convert back to PIL Image
        result_pil_img = upscaled_vertex_img._pil_image  # or .to_pil() if available
        return result_pil_img
