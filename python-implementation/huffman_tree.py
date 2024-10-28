from struct import unpack

class HuffmanTreeNode:
    # Initializing the leaf node
    def __init__(self, length=None, val=None, internal=False):
        self.left = None
        self.right = None
        self.parent = None
        self.val = val
        self.internal = internal
        self.length = length
    
    def __str__(self):
        return f'The value is {self.val} and the length is {self.length}'

class HuffmanTree:
    # Data is the huffman table chunk 
    # showing the character lengths
    # Example: 0 1 5 1 1 1 1 1 1 0 0 0 0 0 0 a b c 
    # The passed data is the after 4 bytes of the section, behind the markers
    def __init__(self, data, root):
        self.data = data
        self.elements = self.create_elements(data)
        self.root = root
        self.codes = {}

    def create_elements(self, data):
        offset = 0
        header, = unpack("B",data[offset:offset+1])
        offset += 1

        # Extract the 16 bytes containing length data
        lengths = unpack("BBBBBBBBBBBBBBBB", data[offset:offset+16]) 
        offset = offset + 16

        elements = []
        for index, val in enumerate(lengths):
            for j in range(val):
                current_val = unpack("B", data[offset:offset+1]) # getting the character
                offset += 1 
                elements.append(HuffmanTreeNode(index + 1, current_val)) # we are using lengths and all leaf nodes have the characters

        # Sorting the elements in descending order in length --> ascending order in frequency
        return sorted(elements, key=lambda item: item.length)
    

    def add_to_tree(self, root, element, pos):
        if root.internal == False:
            return False
        # If position is one, we found the place to insert
        if pos == 1:
            if root.left == None:
                root.left = element
            elif root.right == None:
                root.right = element
            else:
                return False
            return True
        for i in [0,1]:
            internal_node = HuffmanTreeNode(val=i, internal=True)
            if i == 0:
                if root.left == None:
                    root.left = internal_node
                if self.add_to_tree(root.left, element, pos-1) == True:
                    return True
            else:
                if root.right == None:
                    root.right = internal_node
                if self.add_to_tree(root.right, element, pos-1) == True:
                    return True
        return False

    # Returns the codes for characters depending on the huffman tree.
    def decode_tree(self, cur_node, current_bit_string):
        # Base case reached to a leaf node
        if cur_node.val != None:
            self.codes[cur_node.val] = current_bit_string
        # Appending 0 when going left
        if cur_node.left != None:
            self.decode_tree(cur_node.left, current_bit_string + '0')
        # Appending 1 when going right
        if cur_node.right != None:
            self.decode_tree(cur_node.right, current_bit_string + '1')
        return