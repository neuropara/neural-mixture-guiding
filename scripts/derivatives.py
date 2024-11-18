import sympy
from sympy import symbols, diff, factor, latex
from sympy import exp, pi
from sympy import sin, cos, sinh

if __name__ == "__main__":
	sympy.init_printing(use_latex=True, use_unicode=True)
	kappa, theta, phi = symbols("kappa theta varphi", real=True)
	x, y, z = symbols("x y z", real=True)

	# vmf = (kappa / (4 * pi * sinh(kappa))) * exp(kappa * (x * cos(phi) * sin(theta) + y * sin(phi) * sin(theta) + z * cos(theta)))
	vmf = (kappa / (2 * pi * (1 - exp(-2 * kappa)))) * exp(kappa * (x * cos(phi) * sin(theta) + y * sin(phi) * sin(theta) + z * cos(theta) - 1)) 

	df_dkappa = factor(diff(vmf, kappa))
	df_dtheta = factor(diff(vmf, theta))
	df_dvarphi = factor(diff(vmf, phi))

	print("df_dkappa:", latex(df_dkappa))
	print("df_dtheta:", latex(df_dtheta))
	print("df_dphi:", latex(df_dvarphi))
