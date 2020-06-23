import scipy.integrate as integrate

from numpy import sqrt, sin, cos, tan, pi, inf, exp
from algo_all_sym import *

image_path = "test21.png"

(latex, scipy) = get_integral(image_path)

print('LaTeX: ', latex)
print('SciPy: ', scipy)

result = integrate.quad(eval('lambda x: ' + scipy), 2, 3)

print('Integration Result: ', result[0])


