# Compiler
CXX = g++

# Compiler flags
CXXFLAGS = -Ilibjpeg_install/install/include -Wall -Wextra -O2

# Linker flags
LDFLAGS = -Llibjpeg_install/install/lib -ljpeg

# Target
TARGET = decoder

# Source files
SRC = main.cpp src/parser.cpp utils/stream.cpp utils/color.cpp

# Build target
$(TARGET): $(SRC)
	$(CXX) -o $(TARGET) $(SRC) $(CXXFLAGS) $(LDFLAGS)

# Clean target
clean:
	rm -f $(TARGET)