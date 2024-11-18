LDFLAGS = -L./libjpeg/lib -Wl,-rpath,./libjpeg/lib -ljpeg

decoder: main.cpp src/parser.cpp utils/stream.cpp utils/color.cpp
	g++ -o decoder main.cpp src/parser.cpp utils/stream.cpp utils/color.cpp -I./libjpeg/include -L./libjpeg/lib -ljpeg
