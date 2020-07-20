from App.utils import *
from numpy import shape, empty, uint8, log
from typing import Optional, Tuple, List


class PointProcessing():
    def __init__(self, location: str, grayscale: Optional[bool] = False) -> None:
        image, image_shape = read_image(location=location)
        if not grayscale:
            image_grayscale = color_to_gray(image_matrix=image)
        self.image = image_grayscale
        self.image_shape = shape(image_grayscale)
        self.x = image_shape[0]
        self.y = image_shape[1]

    def addition(self, image_two: List[List]) -> List[List]:
        if shape(image_two) != self.image_shape:
            raise Exception("images must have the same dimensions")

        solution_image = empty([self.x, self.y])
        for i in range(self.x):
            for j in range(self.y):
                temp = int(self.image[i][j]) + int(image_two[i][j])
                if temp >= 255:
                    temp = 255
                solution_image[i][j] = temp

        return uint8(solution_image)

    def subtraction(self, image_two: List[List]) -> List[List]:
        if shape(image_two) != self.image_shape:
            raise Exception("images must have the same dimensions")

        solution_image = empty([self.x, self.y])
        for i in range(self.x):
            for j in range(self.y):
                temp = int(self.image[i][j]) - int(image_two[i][j])
                if temp <= 0:
                    temp = 0
                solution_image[i][j] = temp

        return uint8(solution_image)

    def multiplication(self, image_two: List[List]) -> List[List]:
        if shape(image_two) != self.image_shape:
            raise Exception("images must have the same dimensions")

        solution_image = empty([self.x, self.y])
        for i in range(self.x):
            for j in range(self.y):
                temp = int(self.image[i][j]) * int(image_two[i][j])
                if temp >= 255:
                    temp = 255
                solution_image[i][j] = temp

        return uint8(solution_image)

    def division(self, image_two: List[List]) -> List[List]:
        if shape(image_two) != self.image_shape:
            raise Exception("images must have the same dimensions")

        solution_image = empty([self.x, self.y])
        for i in range(self.x):
            for j in range(self.y):
                temp = int(self.image[i][j] / image_two[i][j])
                solution_image[i][j] = temp

        return uint8(solution_image)

    def blending(self, image_two: List[List], blend_factor: float) -> List[List]:
        if shape(image_two) != self.image_shape:
            raise Exception("images must have the same dimensions")

        if blend_factor > 1:
            raise Exception("blend factor must be of range(0, 1)")

        solution_image = empty([self.x, self.y])
        for i in range(self.x):
            for j in range(self.y):
                temp = int(self.image[i][j] * blend_factor) + \
                    int(image_two[i][j] * blend_factor)
                if temp >= 255:
                    temp = 255
                solution_image[i][j] = temp

        return uint8(solution_image)

    def negation(self) -> List[List]:
        solution_image = empty([self.x, self.y])
        for i in range(self.x):
            for j in range(self.y):
                temp = 255 - self.image[i][j]
                solution_image[i][j] = temp

        return uint8(solution_image)

    def thresholding(self, threshold: int) -> List[List]:
        if threshold > 255 or threshold < 0:
            raise Exception("threshold must be of range(0, 255)")

        solution_image = empty([self.x, self.y])
        for i in range(self.x):
            for j in range(self.y):
                temp = 255
                if self.image[i][j] < threshold:
                    temp = 0
                solution_image[i][j] = temp

        return uint8(solution_image)

    def gray_level_slicing(self, slice_range: Tuple[int], background: Optional[bool] = False) -> List[List]:
        if any(slice_range) > 255 or any(slice_range) < 0:
            raise Exception("slice range must be of range(0, 255)")

        if slice_range[0] > slice_range[1]:
            raise Exception("slice range must be of type (min: int, max: int)")

        solution_image = empty([self.x, self.y])
        for i in range(self.x):
            for j in range(self.y):
                if self.image[i][j] >= slice_range[0] and self.image[i][j] <= slice_range[1]:
                    temp = 255
                else:
                    temp = 0
                    if background:
                        temp = self.image[i][j]
                solution_image[i][j] = temp

        return uint8(solution_image)

    def log_transform(self) -> List[List]:
        max_pix = max([max(row) for row in self.image])
        c = (255)/(log(1+max_pix))

        solution_image = empty([self.x, self.y])
        for i in range(self.x):
            for j in range(self.y):
                temp = int(c*log(1+self.image[i][j]))
                solution_image[i][j] = temp

        return uint8(solution_image)
