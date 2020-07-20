import sys
sys.path.insert(0, "E:\Projects\DIP")
from App.utils import *


if __name__ == "__main__":
    im_matrix, im_shape = read_image(location="images\lenna.png")
    print("shape:", im_shape)
    im_grayscale = color_to_gray(im_matrix=im_matrix)
    write_image(im_matrix=im_grayscale)
