from App.utils import *
from App.neighbourhood_processing import NeighbourhoodProcessing
from numpy import shape, empty, zeros, uint8, multiply, dot, median, sqrt
from typing import Optional, Tuple, List


class EdgeDetection():
    def __init__(self, location: str, grayscale: Optional[bool] = False) -> None:
        image, image_shape = read_image(location=location)
        if not grayscale:
            image_grayscale = color_to_gray(image_matrix=image)
        self.image = image_grayscale
        self.image_shape = shape(image_grayscale)
        self.x = image_shape[0]
        self.y = image_shape[1]
        self.masks = {
            "sobel": {
                "x": [
                    [1, 0, -1],
                    [2, 0, -2],
                    [1, 0, -1]
                ],
                "y": [
                    [1, 2, 1],
                    [0, 0, 0],
                    [-1, -2, -1]
                ]
            },
            "prewitt": {
                "x": [
                    [1, 0, -1],
                    [1, 0, -1],
                    [1, 0, -1]
                ],
                "y": [
                    [1, 1, 1],
                    [0, 0, 0],
                    [-1, -1, -1]
                ]
            }
        }

    @staticmethod
    def convolution(image: List[List], filter_x: List[List], filter_y: List[List]) -> List[List]:
        x, y = shape(image)[0], shape(image)[1]
        solution_image = empty([x, y], dtype=uint8)

        padded_image = NeighbourhoodProcessing.pad_image(image)
        for i in range(1, len(image)+1):
            for j in range(1, len(image)+1):
                temp_x = sum([sum(row) for row in multiply(
                    padded_image[i-1:i+2, j-1:j+2], filter_x)])
                temp_y = sum([sum(row) for row in multiply(
                    padded_image[i-1:i+2, j-1:j+2], filter_y)])

                temp = sqrt(pow(temp_x, 2) + pow(temp_y, 2))

                solution_image[i-1][j-1] = int(temp)

        return uint8(solution_image)

    def detect(self, filter: Optional[str] = "sobel") -> List[List]:
        mask_x, mask_y = self.masks[filter]["x"], self.masks[filter]["y"]
        return self.convolution(self.image, mask_x, mask_y)
