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

    # Save the JPEG data to a file
    with open(output_jpeg_path, "wb") as f:
        f.write(jpeg_data)

# Usage example
convert_to_444_jpeg("/Users/ddas/Desktop/Debeshee/ETH/Academics/Sem 3/DPHPC/GPU-JPEG-Decoder/testing/images/5_200x200.jpg", "5_200x200.jpg", quality=95)