import sys
sys.path.insert(0, "E:\Projects\DIP")
from cv2 import resize, imshow, waitKey, destroyAllWindows
from numpy import ones
from App.neighbourhood_processing import *
from App.utils import *

if __name__ == "__main__":
    # pattern_matrix, _ = read_image(location=r"images/test_images/pattern.jpg")
    # pattern_grayscale = color_to_gray(image_matrix=pattern_matrix)
    # pattern = resize(pattern_grayscale, (440, 440))

    np = NeighbourhoodProcessing(r"images/test_images/lenna.png")

    padded_image = np.pad_image(np.image)
    write_image(padded_image, r"images/neighbourhood_processing/padded_image.png")
    low_pass = np.low_pass_filtering()
    write_image(low_pass, r"images/neighbourhood_processing/low_pass.png")
    high_pass = np.high_pass_filtering()
    write_image(high_pass, r"images/neighbourhood_processing/high_pass.png")
    median = np.median_filtering()
    write_image(median, r"images/neighbourhood_processing/median.png")
