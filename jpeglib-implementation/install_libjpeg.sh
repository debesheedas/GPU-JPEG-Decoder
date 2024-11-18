#!/bin/bash

# Set installation directory (adjust as needed)
INSTALL_DIR="/home/dphpc2024_jpeg_1/cfernand/GPU-JPEG-Decoder/jpeglib-implementation/libjpeg"

# Create installation directory
mkdir -p "$INSTALL_DIR"

# Download the latest version of libjpeg
echo "Downloading libjpeg source..."
wget -q --show-progress https://ijg.org/files/jpegsrc.v9e.tar.gz -O jpegsrc.tar.gz

# Extract the source tarball
echo "Extracting libjpeg source..."
tar -xvzf jpegsrc.tar.gz
cd jpeg-9e || { echo "Failed to enter source directory"; exit 1; }

# Configure the build with the specified prefix
echo "Configuring build..."
./configure --prefix="$INSTALL_DIR"

# Build and install
echo "Building and installing libjpeg..."
make -j$(nproc) && make install

# Cleaning up
cd ..
rm -rf jpeg-9e jpegsrc.tar.gz

# Display success message
echo "libjpeg installed to $INSTALL_DIR"