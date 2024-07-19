from sympy import symbols, solve, Eq
from sympy.geometry import Circle, Line, Point
import math
import json
import os

height, width = 480, 640
(h, k) = width/2, height/2 #center of circle
x, y = symbols('x y')
radius = 200
circle_eq = Eq((x - h)**2 + (y - k)**2, radius**2) #find equation for circle
"""
circle's equation: (x-h)^2 + (y-k)^2 = r^2
slope: slope of line going from center of circle to desired point on circle, tan(delta)
line = line going from desired point to center of circle
desired point is intersectino between line quation and circle equation
"""
points = [[], [], [], []]
slope_multiplier = 1
append_idx = (0, 2)
        
print('Loading animations...')
for i in range (0, 2): #process quarters of circle, append points 
    for j in range (90, -1, -1):
                
        if i==0:
            delta  = j
        elif i==1:
            delta = 90 - j
                    
        if delta != 90:
            slope = math.tan(math.radians(delta)) * slope_multiplier
            y_intercept = k - slope*h
            line_eq = Eq(y, slope * x + y_intercept) #find equation for the line
            intersections = solve([circle_eq, line_eq], (x, y))
        else:
            intersections = [(h, k-radius), (h, k+radius)]
                
        intersections = [(int(sol[0]), int(sol[1])) for sol in intersections]
        for l in range (0, 2):
            points[append_idx[l]].append(intersections[l])
                    
    slope_multiplier *= -1
    append_idx = (1, 3) 

dots = []
for quarter in points:
    for intersection in quarter:
        (x, y) = (intersection[0], intersection[1])
        dots.append((x, y))

with open('animation_points.json', 'w') as f:
    json.dump(dots, f)