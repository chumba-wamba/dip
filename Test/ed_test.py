import sys
sys.path.insert(0, "E:\Projects\DIP")
from cv2 import resize, imshow, waitKey, destroyAllWindows
from numpy import ones
from App.edge_detection import *
from App.utils import *


if __name__ == "__main__":
    ed = EdgeDetection(r"images/test_images/lenna.png")

    sobel = ed.detect()
    write_image(sobel, r"images/edge_detection/sobel.png")
    prewitt = ed.detect("prewitt")
    write_image(prewitt, r"images/edge_detection/prewitt.png")

    ed = EdgeDetection(r"images/neighbourhood_processing/low_pass.png")
    laplacian = ed.detect("laplacian")
    write_image(laplacian, r"images/edge_detection/laplacian.png")
    