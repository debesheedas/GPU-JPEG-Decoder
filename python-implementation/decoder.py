from struct import unpack
from huffman_table import HuffmanTable
import math
from huffman_table import Stream

def GetArray(type, l, length):
    """
    A convenience function for unpacking an array from bitstream
    """
    s = ""
    for i in range(length):
        s = s + type
    return list(unpack(s, l[:length]))

def decode_number(code, bits):
    l = 2 ** (code - 1)
    if bits >= l:
        return bits
    else:
        return bits - (2 * l - 1)
    
def clamp(col):
    col = 255 if col > 255 else col
    col = 0 if col < 0 else col
    return int(col)
    
def color_conversion(Y, Cr, Cb):
    """
    Converts Y, Cr and Cb to RGB color space
    """
    R = Cr * (2 - 2 * 0.299) + Y
    B = Cb * (2 - 2 * 0.114) + Y
    G = (Y - 0.114 * B - 0.299 * R) / 0.587
    return (clamp(R + 128), clamp(G + 128), clamp(B + 128))
    
class IDCT: #done

    def __init__(self):
        self.base = [0] * 64
        self.zigzag = [
            [0, 1, 5, 6, 14, 15, 27, 28],
            [2, 4, 7, 13, 16, 26, 29, 42],
            [3, 8, 12, 17, 25, 30, 41, 43],
            [9, 11, 18, 24, 31, 40, 44, 53],
            [10, 19, 23, 32, 39, 45, 52, 54],
            [20, 22, 33, 38, 46, 51, 55, 60],
            [21, 34, 37, 47, 50, 56, 59, 61],
            [35, 36, 48, 49, 57, 58, 62, 63],
        ]
        self.idct_precision = 8
        self.idct_table = [
            [
                (self.NormCoeff(u) * math.cos(((2.0 * x + 1.0) * u * math.pi) / 16.0))
                for x in range(self.idct_precision)
            ]
            for u in range(self.idct_precision)
        ]

    def NormCoeff(self, n):
        if n == 0:
            return 1.0 / math.sqrt(2.0)
        else:
            return 1.0

    def rearrange_using_zigzag(self):
        for x in range(8):
            for y in range(8):
                self.zigzag[x][y] = self.base[self.zigzag[x][y]]
        return self.zigzag

    def perform_IDCT(self):
        out = [list(range(8)) for i in range(8)]

        for x in range(8):
            for y in range(8):
                local_sum = 0
                for u in range(self.idct_precision):
                    for v in range(self.idct_precision):
                        local_sum += (
                            self.zigzag[v][u]
                            * self.idct_table[u][x]
                            * self.idct_table[v][y]
                        )
                out[y][x] = local_sum // 4 # Adjust scale by dividing by 4.0
        self.base = out # Update the base with the computed values

from huffman_tree import HuffmanTree, HuffmanTreeNode
class JPEG:
    def __init__(self, image_file):
        f = open(image_file, "rb")
        self.img_data = f.read()
        self.quantMapping = []
        self.quant = {}
        self.width = 0
        self.height = 0
        self.huffman_tables = {}

        # Extracting the data parts of the JPEG image.
        self.data_chunks = self.extract_chunks()
        self.decode()

    # 0xFF is a special character, if it is in the image it is followed by 0x00 for distinction. 
    # When decoding, we need to remove these 00's.
    def remove_byte_stuffing(self, data):
        datapro = []
        i = 0
        while(True):
            b,bnext = unpack("BB",data[i:i+2])  
            if (b == 0xff):
                if (bnext != 0x00):
                    break
                datapro.append(data[i])
                i+=2
            else:
                datapro.append(data[i])
                i+=1
        return datapro,i
    

    def decodeHuffman(self, data):
        offset = 0
        (header,) = unpack("B", data[offset : offset + 1])
        print(header, header & 0x0F, (header >> 4) & 0x0F)
        offset += 1

        lengths = GetArray("B", data[offset : offset + 16], 16)
        offset += 16

        elements = []
        for i in lengths:
            elements += GetArray("B", data[offset : offset + i], i)
            offset += i

        hf = HuffmanTable(data)
        hf.GetHuffmanBits(lengths, elements)
        self.huffman_tables[header] = hf
        data = data[offset:]

    def build_matrix(self, st, idx, quant, olddccoeff):
        i = IDCT()
        code = self.huffman_tables[0 + idx].GetCode(st)
        bits = st.GetBitN(code)
        dccoeff = decode_number(code, bits) + olddccoeff

        i.base[0] = (dccoeff) * quant[0]
        l = 1
        while l < 64:
            code = self.huffman_tables[16 + idx].GetCode(st)
            if code == 0:
                break
            if code > 15:
                l += code >> 4
                code = code & 0x0F

            bits = st.GetBitN(code)

            if l < 64:
                coeff = decode_number(code, bits)
                i.base[l] = coeff * quant[l]
                l += 1

        i.rearrange_using_zigzag()
        i.perform_IDCT()

        return i, dccoeff
    
    def start_of_scan(self, data, header_len):
        # Remove byte stuffing from the data after the header
        data, len_chunk = self.remove_byte_stuffing(data[header_len:])
        
        # Initialize a stream from the processed data
        stream = Stream(data)

        # Initialize previous DC coefficients for luminance and chrominance
        prev_luminance_dc = 0
        prev_chrominance_cb_dc = 0
        prev_chrominance_cr_dc = 0

        # Loop through each 8x8 block in the image
        for block_y in range(self.height // 8):
            for block_x in range(self.width // 8):
                # Build matrices for luminance and chrominance components
                matL, prev_luminance_dc = self.build_matrix(stream, 0, self.quant[self.quantMapping[0]], prev_luminance_dc)
                # matCr, prev_chrominance_cr_dc = self.build_matrix(stream, 1, self.quant[self.quantMapping[1]], prev_chrominance_cr_dc)
                # matCb, prev_chrominance_cb_dc = self.build_matrix(stream, 1, self.quant[self.quantMapping[2]], prev_chrominance_cb_dc)

                # # Draw the matrix for the current block
                # self.draw_matrix(block_x, block_y, matL.base, matCb.base, matCr.base)
                pass

        # Return the total length of processed data including the header
        return len_chunk + header_len
    
    def parse_sof(self, data):
        hdr, self.height, self.width, components = unpack(">BHHB",data[0:6])
        print("size %ix%i" % (self.width,  self.height))

        for i in range(components):
            id, samp, QtbId = unpack("BBB",data[6+i*3:9+i*3])
            self.quantMapping.append(QtbId)      

    def extract_chunks(self):
        data_chunks = {}
        data = self.img_data
        while True:
            (marker,) = unpack(">H", data[0:2])
            if marker == 0xFFD8:
                data = data[2:]
            elif marker == 0xFFD9:
                return data_chunks
            else:
                (len_chunk,) = unpack(">H", data[2:4]) # Length of the segment
                len_chunk += 2 # Adding the inital FFDA part for complete header length
                chunk = data[4:len_chunk]
                if marker == 0xFFC4:
                    if 'Huffman_Tables' not in data_chunks:
                        data_chunks['Huffman_Tables'] = []
                    data_chunks['Huffman_Tables'].append(chunk)
                    self.decodeHuffman(chunk)
                elif marker == 0xFFDB:
                    if 'Quant_Tables' not in data_chunks:
                        data_chunks["Quant_Tables"] = []
                    hdr, = unpack("B",chunk[0:1])
                    print(hdr, ' is header')
                    self.quant[hdr] = GetArray("B", chunk[1 : 1 + 64], 64)
                    data_chunks["Quant_Tables"].append(chunk)
                elif marker == 0xFFC0:
                    data_chunks['Start_Of_Frame'] = chunk
                    self.parse_sof(data_chunks['Start_Of_Frame'])
                elif marker == 0xFFDA:
                    len_chunk = self.start_of_scan(data, len_chunk)
                    data_chunks['Image'] = data
                data = data[len_chunk:]
            if len(data) == 0:
                break
        
    def print_8x8_matrix(self, binary_data):
        # Unpack the binary data into 64 unsigned 8-bit integers (64 "B" format characters)
        unpacked_data = unpack("64B", binary_data)
        
        # Format and print the 8x8 matrix
        print("8x8 Matrix (Decimal Values):")
        for i in range(8):
            # Print 8 values per row (i * 8 to (i+1) * 8)
            row = unpacked_data[i * 8:(i + 1) * 8]
            print(" ".join(f"{value:2d}" for value in row)) 


    def decode(self):
        # Step 1: Extracting Huffman Tables
        root = HuffmanTreeNode(internal=True)
        tree =  HuffmanTree(self.data_chunks['Huffman_Tables'][0], root)
        for item in tree.elements:
            tree.add_to_tree(root, item, item.length)
        tree.decode_tree(tree.root, '')
        for item,val in tree.codes.items():
            print(item, ' ' , val)

        # Step 2: Extracting Quantization Values
        print(self.data_chunks['Quant_Tables'][0][1:]) # 0 for luminance
        print(len(self.data_chunks['Quant_Tables'][0][1:]))
        print(self.print_8x8_matrix(self.data_chunks['Quant_Tables'][0][1:]))
        print(self.data_chunks['Quant_Tables'][1][1:]) # 1 for others

if __name__ == "__main__":
    img = JPEG('profile.jpg')
    img.decode()    