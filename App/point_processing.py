from App.utils import *
from numpy import shape, empty, uint8
from typing import Optional, Tuple, List


class PointProcessing():
    def __init__(self, location: str, grayscale: Optional[bool] = False) -> None:
        im_one, im_shape = read_image(location=location)
        if not grayscale:
            im_grayscale = color_to_gray(im_matrix=im_one)
        self.im_one = im_grayscale
        self.im_shape = shape(im_grayscale)
        self.x = im_shape[0]
        self.y = im_shape[1]

    def addition(self, im_two: List[List]) -> List[List]:
        if shape(im_two) != self.im_shape:
            raise Exception("images must have the same dimensions")

        im_solution = empty([self.x, self.y])
        for i in range(self.x):
            for j in range(self.y):
                temp = int(self.im_one[i][j]) + int(im_two[i][j])
                if temp >= 255:
                    temp = 255
                im_solution[i][j] = temp

        return uint8(im_solution)

    def subtraction(self, im_two: List[List]) -> List[List]:
        if shape(im_two) != self.im_shape:
            raise Exception("images must have the same dimensions")

        im_solution = empty([self.x, self.y])
        for i in range(self.x):
            for j in range(self.y):
                temp = int(self.im_one[i][j]) - int(im_two[i][j])
                if temp <= 0:
                    temp = 0
                im_solution[i][j] = temp

        return uint8(im_solution)

    def multiplication(self, im_two: List[List]) -> List[List]:
        if shape(im_two) != self.im_shape:
            raise Exception("images must have the same dimensions")

        im_solution = empty([self.x, self.y])
        for i in range(self.x):
            for j in range(self.y):
                temp = int(self.im_one[i][j]) * int(im_two[i][j])
                if temp >= 255:
                    temp = 255
                im_solution[i][j] = temp

        return uint8(im_solution)

    def division(self, im_two: List[List]) -> List[List]:
        if shape(im_two) != self.im_shape:
            raise Exception("images must have the same dimensions")

        im_solution = empty([self.x, self.y])
        for i in range(self.x):
            for j in range(self.y):
                temp = int(self.im_one[i][j] / im_two[i][j])
                im_solution[i][j] = temp

        return uint8(im_solution)

    def blending(self, im_two: List[List], blend_factor: float) -> List[List]:
        if shape(im_two) != self.im_shape:
            raise Exception("images must have the same dimensions")

        if blend_factor > 1:
            raise Exception("blend factor must be of range(0, 1)")

        im_solution = empty([self.x, self.y])
        for i in range(self.x):
            for j in range(self.y):
                temp = int(self.im_one[i][j] * blend_factor) + \
                    int(im_two[i][j] * blend_factor)
                if temp >= 255:
                    temp = 255
                im_solution[i][j] = temp

        return uint8(im_solution)

    def negation(self):
        im_solution = empty([self.x, self.y])
        for i in range(self.x):
            for j in range(self.y):
                temp = 255 - self.im_one[i][j]
                im_solution[i][j] = temp

        return uint8(im_solution)

    def thresholding(self, threshold: int):
        if threshold > 255 or threshold < 0:
            raise Exception("threshold must be of range(0, 255)")

        im_solution = empty([self.x, self.y])
        for i in range(self.x):
            for j in range(self.y):
                temp = 255
                if self.im_one[i][j] < threshold:
                    temp = 0
                im_solution[i][j] = temp

        return uint8(im_solution)
