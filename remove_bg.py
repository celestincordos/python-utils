import argparse
from pathlib import Path

import cv2
import numpy as np
from PIL import Image


def remove_background(image: Image.Image) -> Image.Image:
    """
    Removes background from a PIL image using OpenCV thresholding.
    Returns a new PIL Image with transparency.
    """
    # Convert PIL -> NumPy (OpenCV uses BGR)
    cv_image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)

    # Convert to grayscale
    gray = cv2.cvtColor(cv_image, cv2.COLOR_BGR2GRAY)

    # Thresholding (tune 240 if background isn't pure white)
    _, thresh = cv2.threshold(gray, 240, 255, cv2.THRESH_BINARY)

    # Invert and create mask
    mask = cv2.bitwise_not(thresh)

    # Apply mask to original image
    result = cv2.bitwise_and(cv_image, cv_image, mask=mask)

    # Convert result to BGRA (add alpha channel using mask)
    result = cv2.cvtColor(result, cv2.COLOR_BGR2BGRA)
    result[:, :, 3] = mask  # use mask as alpha

    # Convert back NumPy -> PIL
    return Image.fromarray(cv2.cvtColor(result, cv2.COLOR_BGRA2RGBA))


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Remove the background from an image using rembg."
    )
    parser.add_argument("input", help="Path to the input image")
    parser.add_argument(
        "-o", "--output", help="Path to save the output image (optional)"
    )

    args = parser.parse_args()

    input_path = Path(args.input)
    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    # Open input image
    input_image = Image.open(input_path)

    # Process background removal
    output_image = remove_background(input_image)

    # Determine output path
    output_path = args.output or input_path.with_name(f"{input_path.stem}_no_bg.png")

    # Save result
    output_image.save(output_path)
    print(f"✅ Background removed: {output_path}")


if __name__ == "__main__":
    main()
