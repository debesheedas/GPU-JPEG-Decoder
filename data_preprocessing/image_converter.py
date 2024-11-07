import argparse
from PIL import Image
import numpy as np
import simplejpeg

def convert_to_444_jpeg(input_image_path, output_jpeg_path, quality=95):
    # Load the image using Pillow and convert it to RGB if it's not
    image = Image.open(input_image_path).convert("RGB")

    # Convert the image to a numpy array
    image_data = np.array(image)

    # Encode to JPEG with 4:4:4 chroma subsampling
    jpeg_data = simplejpeg.encode_jpeg(
        image_data,
        quality=quality,
        colorspace="RGB",
        colorsubsampling='444'  # Use 4:4:4 chroma subsampling
    )

    with open(output_jpeg_path, "wb") as f:
        f.write(jpeg_data)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert an image to 4:4:4 JPEG format.")
    parser.add_argument("input_image_path", type=str, help="Path to the input image file")
    parser.add_argument("output_jpeg_path", type=str, help="Path to save the output JPEG file")
    parser.add_argument("--quality", type=int, default=95, help="JPEG quality (default: 95)")

    args = parser.parse_args()

    convert_to_444_jpeg(args.input_image_path, args.output_jpeg_path, quality=args.quality)