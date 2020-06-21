import scipy.integrate as integrate

from numpy import sqrt, sin, cos, tan, pi
from algo_all_sym import *

image_path = "test6.png"

(latex, scipy_rect) = get_integral(image_path)

print('LaTeX: ', latex)
print('SciPy: ', scipy_rect)

result = integrate.quad(eval('lambda x: ' + scipy_rect), 2, 3)

print('Integration Result: ', result[0])


