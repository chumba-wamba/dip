from cv2 import imread, imwrite, cvtColor, COLOR_BGR2GRAY
from numpy import shape
from typing import Optional, Tuple, List


def read_image(location: str) -> Tuple[List[List[List]], Tuple]:
    image_matrix = imread(location)
    image_shape = shape(image_matrix)
    return image_matrix, image_shape


def write_image(image_matrix: List[List[List]], location: Optional[str] = "test_img.png") -> None:
    imwrite(location, image_matrix)


def color_to_gray(image_matrix: List[List[List]]) -> List[List]:
    im_grayscale = cvtColor(image_matrix, COLOR_BGR2GRAY)
    return im_grayscale
