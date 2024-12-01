import sys
from pathlib import Path

def process_ppm(input_path, output_dir):
    with open(input_path, 'rb') as f:
        # Read PPM header
        header = f.readline().decode().strip()
        if header != "P6":
            raise ValueError("Input is not a binary PPM file.")
        
        # Skip comments
        while True:
            line = f.readline().decode().strip()
            if not line.startswith("#"):
                break
        
        # Read dimensions and max value
        width, height = map(int, line.split())
        max_val = int(f.readline().decode().strip())
        
        # Read binary pixel data
        raw_data = list(f.read())
    
    # Extract RGB channels
    r = raw_data[0::3]
    g = raw_data[1::3]
    b = raw_data[2::3]

    # Write to the `.array` file
    output_path = Path(output_dir) / (Path(input_path).stem + ".array")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, 'w') as outfile:
        # Write height and width
        outfile.write(f"{height} {width}\n")
        # Write R, G, B channels
        outfile.write(" ".join(map(str, r)) + "\n")
        outfile.write(" ".join(map(str, g)) + "\n")
        outfile.write(" ".join(map(str, b)) + "\n")

    print(f"Output written to {output_path}")

# Usage example
if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python script.py <input.ppm> <output_directory>")
    else:
        process_ppm(sys.argv[1], sys.argv[2])