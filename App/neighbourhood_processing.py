from App.utils import *
from numpy import shape, empty, zeros, uint8, multiply, dot, median
from typing import Optional, Tuple, List


class NeighbourhoodProcessing():
    def __init__(self, location: str, grayscale: Optional[bool] = False) -> None:
        image, image_shape = read_image(location=location)
        if not grayscale:
            image_grayscale = color_to_gray(image_matrix=image)
        self.image = image_grayscale
        self.image_shape = shape(image_grayscale)
        self.x = image_shape[0]
        self.y = image_shape[1]

    @staticmethod
    def pad_image(image: List[List]) -> List[List]:
        x, y = shape(image)[0], shape(image)[1]
        padded_image = zeros([x+2, y+2], dtype=uint8)
        for i in range(1, x+1):
            for j in range(1, y+1):
                padded_image[i][j] = image[i-1][j-1]
        return padded_image

    @staticmethod
    def convolution(image: List[List], filter: List[List]) -> List[List]:
        x, y = shape(image)[0], shape(image)[1]
        solution_image = empty([x, y], dtype=uint8)

        padded_image = NeighbourhoodProcessing._pad_image(image)
        for i in range(1, len(image)+1):
            for j in range(1, len(image)+1):
                temp = sum([sum(row) for row in multiply(
                    padded_image[i-1:i+2, j-1:j+2], filter)])
                solution_image[i-1][j-1] = int(temp)

        return uint8(solution_image)

    @classmethod
    def _pad_image(cls, image):
        return cls.pad_image(image)

    def low_pass_filtering(self):
        low_pass_mask = [
            [1/9, 1/9, 1/9],
            [1/9, 1/9, 1/9],
            [1/9, 1/9, 1/9]
        ]
        low_pass_image = self.convolution(
            image=self.image, filter=low_pass_mask)
        return low_pass_image

    def high_pass_filtering(self):
        low_pass_mask = [
            [-1/9, -1/9, -1/9],
            [-1/9, 8/9, -1/9],
            [-1/9, -1/9, -1/9]
        ]
        low_pass_image = self.convolution(
            image=self.image, filter=low_pass_mask)
        return low_pass_image

    def median_filtering(self):
        x, y = shape(self.image)[0], shape(self.image)[1]
        low_pass_image = empty([x, y], dtype=uint8)

        padded_image = self.pad_image(self.image)
        for i in range(1, len(self.image)+1):
            for j in range(1, len(self.image)+1):
                temp = median(padded_image[i-1:i+2, j-1:j+2])
                low_pass_image[i-1][j-1] = int(temp)

        return uint8(low_pass_image)
