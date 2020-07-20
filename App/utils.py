from cv2 import imread, imwrite, cvtColor, COLOR_BGR2GRAY
from numpy import shape
from typing import Optional, Tuple, List


def read_image(location: str) -> Tuple[List[List[List]], Tuple]:
    im_matrix = imread(location)
    im_shape = shape(im_matrix)
    return im_matrix, im_shape


def write_image(im_matrix: List[List[List]], location: Optional[str] = "test_img.png") -> None:
    imwrite(location, im_matrix)


def color_to_gray(im_matrix: List[List[List]]) -> List[List]:
    im_grayscale = cvtColor(im_matrix, COLOR_BGR2GRAY)
    return im_grayscale
