import scipy.integrate as integrate
import scipy.special as special

from numpy import sqrt, sin, cos, tan, pi, inf, exp, log, log10
from algo_all_sym import *

image_path = "Test Images/test27.png"

(latex, scipy) = get_integral(image_path)

print('LaTeX: ', latex)
print('SciPy: ', scipy)

result = integrate.quad(eval('lambda x: ' + scipy), 2, 3)

print('Integration Result: ', result[0])


