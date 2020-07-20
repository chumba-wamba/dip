import sys
sys.path.insert(0, "E:\Projects\DIP")
from cv2 import resize
from numpy import ones
from App.point_processing import *
from App.utils import *


if __name__ == "__main__":
    pattern_matrix, _ = read_image(location=r"images/test_images/pattern.jpg")
    pattern_grayscale = color_to_gray(im_matrix=pattern_matrix)
    pattern = resize(pattern_grayscale, (440, 440))

    pp = PointProcessing(r"images/test_images/lenna.png")
    
    addition = pp.addition(pattern)
    write_image(addition, r"images/point_processing/addition.png")
    subtraction = pp.subtraction(pattern)
    write_image(subtraction, r"images/point_processing/subtraction.png")
    multiplication = pp.multiplication(pattern)
    write_image(multiplication, r"images/point_processing/multiplication.png")
    division = pp.division(pattern)
    write_image(division, r"images/point_processing/division.png")
    blending = pp.blending(pattern, 0.5)
    write_image(blending, r"images/point_processing/blending.png")
    negation = pp.negation()
    write_image(negation, r"images/point_processing/negation.png")
    thresholding = pp.thresholding(128)
    write_image(thresholding, r"images/point_processing/thresholding.png")
    