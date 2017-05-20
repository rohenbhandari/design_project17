from sympy import Function, symbols, dsolve
x, y = symbols('x y')
expr = x + y - 1
expr.subs(x, 2)
print expr.subs([(x, 1), (y, 2)])