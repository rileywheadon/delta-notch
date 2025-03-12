from sympy import solve, symbols

g, gI, kT, n, nm, n0, de = symbols('g gI kT n nm n0 de')
iStar = (kT * n * de) / gI
sol = solve((1 / (g + (kT * de))) * ((nm * (iStar**2)) / ((n0**2) + (iStar**2))) - n, n)

print(sol)





