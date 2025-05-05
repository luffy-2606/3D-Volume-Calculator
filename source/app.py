from flask import Flask, render_template, request, jsonify
import numpy as np
import sympy as sp
import plotly.graph_objects as go
from sklearn.ensemble import RandomForestClassifier
import json
from typing import Tuple, Dict, Optional, Union

app = Flask(__name__)

def parse_equation(equation_str: str) -> Tuple[Optional[sp.Expr], Optional[str]]:
    """
    Parse a string equation into a SymPy expression and determine its type.
    Returns the parsed expression and the type of region.
    """
    x, y, z = sp.symbols('x y z')
    try:
        # Replace common mathematical notation with Python notation
        equation_str = equation_str.replace('^', '**')
        # Clean up the input string
        equation_str = equation_str.replace(' ', '')  # Remove spaces
        # Handle equations with = sign: move all terms to one side
        if '=' in equation_str:
            left_side, right_side = equation_str.split('=', 1)
            equation_str = f"({left_side})-({right_side})"
        # Parse the expression
        expr = sp.sympify(equation_str)
        expr = sp.expand(expr)
        # Determine region type using polynomial features
        poly = sp.Poly(expr, x, y, z)
        monoms = poly.monoms()
        # Determine which variables have squared terms
        sq_vars = set()
        for monom in monoms:
            if monom[0] == 2: sq_vars.add('x')
            if monom[1] == 2: sq_vars.add('y')
            if monom[2] == 2: sq_vars.add('z')
        # Check if linear z term present
        has_linear_z = poly.coeff_monomial((0, 0, 1)) != 0
        # Classify shape
        if sq_vars == {'x', 'y', 'z'}:
            region_type = 'sphere'
        elif len(sq_vars) == 2:
            region_type = 'paraboloid' if has_linear_z else 'cylinder'
        elif len(sq_vars) == 1:
            region_type = 'paraboloid' if has_linear_z else 'cylinder'
        else:
            region_type = 'plane'
        # Ensure orientation for paraboloid and plane
        if region_type in ['paraboloid', 'plane']:
            coeff_z = poly.coeff_monomial((0, 0, 1))
            if coeff_z is not None and coeff_z < 0:
                expr = -expr
        return expr, region_type
    except Exception as e:
        print(f"Error parsing equation: {e}")
        return None, None

from sklearn.ensemble import RandomForestClassifier
import numpy as np

def find_coordinate_system(equation_str: str) -> str:
    # Updated training dataset
    X_train = np.array([
        [1, 2, 3], [1, 1, 1], [2, 2, 4], [3, 1, 5], [5, 6, 7],
        [2, 3, 4], [1, 4, 2], [5, 5, 5], [6, 6, 8], [7, 7, 9],
        [1, 3, 2], [2, 5, 6], [3, 2, 3], [4, 3, 5], [6, 4, 6],
        [7, 3, 8], [8, 2, 7], [2, 4, 5], [3, 5, 7], [4, 4, 6],
        [5, 3, 4], [2, 2, 2], [6, 6, 7], [7, 6, 8], [8, 5, 9],
        [5, 5, 6], [4, 2, 4], [3, 1, 3], [8, 7, 6], [7, 5, 4],
        [6, 4, 5], [5, 6, 7], [3, 3, 2], [2, 3, 5], [4, 5, 6],
        [6, 7, 8], [3, 4, 5], [2, 5, 4], [6, 6, 6], [7, 4, 3],
        [4, 4, 5], [5, 3, 2], [7, 6, 6], [5, 7, 8], [8, 7, 7],
        [9, 9, 9], [3, 3, 3], [6, 6, 5], [5, 4, 6], [4, 6, 7],
        [8, 8, 7], [7, 7, 6], [3, 5, 6], [1, 3, 4], [4, 3, 2],
        [9, 9, 5], [8, 6, 6], [2, 2, 6], [6, 5, 7], [7, 8, 5],
        [3, 4, 4], [2, 3, 6], [4, 6, 8], [7, 8, 9], [1, 2, 5],
        [3, 2, 4], [5, 6, 5], [9, 8, 7], [8, 7, 6], [4, 3, 4],
        [7, 9, 6], [6, 4, 7], [5, 5, 7], [9, 6, 8], [8, 5, 9],
        [7, 7, 8], [2, 5, 6], [9, 9, 6], [4, 5, 5], [3, 5, 7],
        [1, 2, 2], [4, 3, 6], [8, 5, 4], [7, 7, 7], [6, 6, 4],
        [1, 1, 2], [4, 1, 1], [5, 2, 3], [7, 8, 4], [6, 9, 3],
        [9, 5, 2], [3, 8, 6], [7, 2, 9], [2, 8, 3], [6, 3, 1],
        [4, 2, 5], [1, 5, 8], [5, 6, 2], [8, 4, 7], [6, 7, 2],
        [2, 6, 4], [3, 7, 1], [8, 2, 5], [5, 3, 9], [7, 5, 2],
        [9, 3, 8], [4, 6, 2], [3, 4, 7], [6, 2, 5]
    ])

    y_train = np.array([
        'Cylindrical', 'Spherical', 'Cartesian', 'Cylindrical', 'Cartesian',
        'Cylindrical', 'Spherical', 'Cartesian', 'Cylindrical', 'Spherical',
        'Cartesian', 'Cylindrical', 'Spherical', 'Cylindrical', 'Cartesian',
        'Cylindrical', 'Spherical', 'Cartesian', 'Cylindrical', 'Spherical',
        'Cartesian', 'Cylindrical', 'Spherical', 'Cylindrical', 'Cartesian',
        'Cylindrical', 'Spherical', 'Cartesian', 'Cylindrical', 'Spherical',
        'Cartesian', 'Cylindrical', 'Spherical', 'Cartesian', 'Cylindrical',
        'Spherical', 'Cartesian', 'Cylindrical', 'Spherical', 'Cartesian',
        'Cylindrical', 'Spherical', 'Cartesian', 'Cylindrical', 'Spherical',
        'Cartesian', 'Cylindrical', 'Spherical', 'Cartesian', 'Cylindrical',
        'Spherical', 'Cartesian', 'Cylindrical', 'Spherical', 'Cartesian',
        'Cylindrical', 'Spherical', 'Cartesian', 'Cylindrical', 'Spherical',
        'Cartesian', 'Cylindrical', 'Spherical', 'Cartesian', 'Cylindrical',
        'Spherical', 'Cartesian', 'Cylindrical', 'Spherical', 'Cartesian',
        'Cylindrical', 'Spherical', 'Cartesian', 'Cylindrical', 'Spherical',
        'Cartesian', 'Cylindrical', 'Spherical', 'Cartesian', 'Cylindrical',
        'Spherical', 'Cartesian', 'Cylindrical', 'Spherical', 'Cartesian',
        'Cylindrical', 'Spherical', 'Cartesian', 'Cylindrical', 'Spherical',
        'Cylindrical', 'Spherical', 'Cartesian', 'Cylindrical', 'Spherical',
        'Cartesian', 'Cylindrical', 'Spherical', 'Cartesian', 'Cylindrical',
        'Spherical', 'Cartesian', 'Cylindrical', 'Spherical', 'Cartesian',
        'Cylindrical', 'Spherical', 'Cartesian', 'Cylindrical', 'Spherical'
    ])

    # Train the classifier
    clf = RandomForestClassifier(n_estimators=100, random_state=42)
    clf.fit(X_train, y_train)

    # Preprocess the equation string
    equation_str = equation_str.replace(' ', '').replace('^', '**').lower()

    # Feature extraction from equation
    has_x2 = 'x**2' in equation_str
    has_y2 = 'y**2' in equation_str
    has_z2 = 'z**2' in equation_str
    has_z = 'z' in equation_str

    # Simple feature vector: [xy^2 no z^2, xyz^2, only linear z]
    features = [
        int(has_x2 and has_y2 and not has_z2),
        int(has_x2 and has_y2 and has_z2),
        int(has_z and not (has_x2 or has_y2 or has_z2))
    ]

    return clf.predict([features])[0]


def generate_points(bounds: dict, n_points: int = 100000) -> np.ndarray:
    """
    Generate random points within the specified bounds for Monte Carlo integration.
    """
    points = np.random.uniform(
        low=[bounds['x'][0], bounds['y'][0], bounds['z'][0]],
        high=[bounds['x'][1], bounds['y'][1], bounds['z'][1]],
        size=(n_points, 3)
    )
    return points

def estimate_volume(expr1: sp.Expr, type1: str, expr2: sp.Expr, type2: str, 
                    bounds: dict, n_points: int = 1000000) -> float:
    """
    Estimate the volume of intersection using Monte Carlo integration.
    """
    # Check if the regions are identical
    if type1 == type2 and expr1.equals(expr2):
        if type1 == 'cylinder':
            print("\nNote: These are identical cylinders. The intersection volume is infinite.")
            return float('inf')
        elif type1 == 'sphere':
            try:
                poly = sp.Poly(expr1.expand(), *sp.symbols('x y z'))
                constant_term = poly.coeff_monomial((0, 0, 0))
                r = float(sp.sqrt(-float(constant_term)))
                volume = (4/3) * np.pi * r**3
                print("\nNote: These are identical spheres. Using exact volume formula.")
                return volume
            except Exception as e:
                print(f"\nWarning: Could not calculate exact sphere volume: {e}")
        elif type1 == 'paraboloid':
            print("\nNote: These are identical paraboloids. The intersection volume is infinite.")
            return float('inf')
    points = generate_points(bounds, n_points)
    x_vals = points[:, 0]
    y_vals = points[:, 1]
    z_vals = points[:, 2]
    func1 = sp.lambdify((sp.symbols('x'), sp.symbols('y'), sp.symbols('z')), expr1, 'numpy')
    func2 = sp.lambdify((sp.symbols('x'), sp.symbols('y'), sp.symbols('z')), expr2, 'numpy')
    try:
        if type1 in ['sphere', 'cylinder']:
            in_region1 = func1(x_vals, y_vals, z_vals) <= 0
        else:
            in_region1 = func1(x_vals, y_vals, z_vals) >= 0
        if type2 in ['sphere', 'cylinder']:
            in_region2 = func2(x_vals, y_vals, z_vals) <= 0
        else:
            in_region2 = func2(x_vals, y_vals, z_vals) >= 0
        count = np.sum(in_region1 & in_region2)
        volume = (count / n_points) * np.prod([
            bounds['x'][1] - bounds['x'][0],
            bounds['y'][1] - bounds['y'][0],
            bounds['z'][1] - bounds['z'][0]
        ])
        return volume
    except Exception as e:
        print(f"\nError in volume calculation: {e}")
        return 0.0

def get_optimal_bounds(expr1: sp.Expr, type1: str, expr2: sp.Expr, type2: str) -> dict:
    """
    Determine optimal bounds for the integration based on the region types.
    """
    x, y, z = sp.symbols('x y z')
    if type1 == 'cylinder' and type2 == 'sphere':
        try:
            cylinder_terms = expr1.expand().as_coefficients_dict()
            r1 = float(sp.sqrt(-float(cylinder_terms[1])))
            sphere_terms = expr2.expand().as_coefficients_dict()
            r2 = float(sp.sqrt(-float(sphere_terms[1])))
            r = min(r1, r2)
            return {
                'x': [-r, r],
                'y': [-r, r],
                'z': [-r2, r2]
            }
        except Exception as e:
            print(f"Warning: Using default bounds. Error: {e}")
    elif (type1 == 'paraboloid' and type2 == 'cylinder') or (type2 == 'paraboloid' and type1 == 'cylinder'):
        try:
            if type1 == 'cylinder':
                expr1, expr2 = expr2, expr1
                type1, type2 = type2, type1
            cylinder_terms = expr2.expand().as_coefficients_dict()
            r = float(sp.sqrt(-float(cylinder_terms[1])))
            max_z = r * r
            return {
                'x': [-r, r],
                'y': [-r, r],
                'z': [0, max_z]
            }
        except Exception as e:
            print(f"Warning: Using default bounds. Error: {e}")
    elif (type1 == 'paraboloid' and type2 == 'sphere') or (type1 == 'sphere' and type2 == 'paraboloid'):
        try:
            if type1 == 'sphere':
                sphere_expr, parab_expr = expr1, expr2
            else:
                sphere_expr, parab_expr = expr2, expr1
            sphere_poly = sp.Poly(sphere_expr.expand(), x, y, z)
            constant_term = sphere_poly.coeff_monomial((0, 0, 0))
            R = float(sp.sqrt(-float(constant_term)))
            return {
                'x': [-R, R],
                'y': [-R, R],
                'z': [0, R]
            }
        except Exception as e:
            print(f"Warning: Using default bounds for sphere-paraboloid. Error: {e}")
    return {
        'x': [-10.0, 10.0],
        'y': [-10.0, 10.0],
        'z': [-10.0, 10.0]
    }

def get_optimal_volume_coordinates(type1: str, type2: str) -> str:
    """
    Determine the most suitable coordinate system for calculating the volume
    based on the types of regions being intersected.
    """
    types = sorted([type1, type2])
    if types == ['cylinder', 'cylinder']:
        return "Cylindrical (r, θ, z) - due to circular symmetry in both surfaces"
    elif types == ['cylinder', 'sphere']:
        return "Cylindrical (r, θ, z) - due to circular symmetry and simpler bounds"
    elif types == ['sphere', 'sphere']:
        return "Spherical (ρ, φ, θ) - natural choice for spherical surfaces"
    elif types == ['paraboloid', 'sphere']:
        return "Cylindrical (r, θ, z) - simpler bounds than spherical coordinates"
    elif types == ['cylinder', 'paraboloid']:
        return "Cylindrical (r, θ, z) - natural choice for both surfaces"
    elif types == ['paraboloid', 'paraboloid']:
        return "Cylindrical (r, θ, z) - simplifies the integration"
    elif 'plane' in types and 'sphere' in types:
        return "Spherical (ρ, φ, θ) - optimal for sphere-plane intersection"
    elif 'plane' in types:
        return "Cartesian (x, y, z) - simplest for planes"
    else:
        return "Cartesian (x, y, z) - default choice"

def create_3d_plot(expr1, type1, expr2, type2, bounds):
    """Create a Plotly figure for the 3D regions."""
    import numpy as np
    x, y, z = sp.symbols('x y z')
    
    data = []
    # Color mapping for different types
    color_map = {
        'sphere': 'Reds',
        'cylinder': 'Greens',
        'paraboloid': 'Viridis',
        'plane': 'Blues'
    }
    alt_colors = {
        'sphere': 'Blues',
        'cylinder': 'Oranges',
        'paraboloid': 'Cividis',
        'plane': 'OrRd'
    }
    color1 = color_map.get(type1, 'Reds')
    color2 = color_map.get(type2, 'Reds')
    if color1 == color2:
        color2 = alt_colors.get(type2, 'OrRd')
    # Helper to create a sphere surface
    def add_sphere(expr, name, color):
        poly = sp.Poly(expr.expand(), x, y, z)
        const_term = poly.coeff_monomial((0, 0, 0))
        R = float(sp.sqrt(-float(const_term)))
        phi_vals = np.linspace(0, 2*np.pi, 50)
        theta_vals = np.linspace(0, np.pi, 50)
        Phi, Theta = np.meshgrid(phi_vals, theta_vals)
        Xs = R * np.sin(Theta) * np.cos(Phi)
        Ys = R * np.sin(Theta) * np.sin(Phi)
        Zs = R * np.cos(Theta)
        surface = go.Surface(
            x=Xs.tolist(),
            y=Ys.tolist(),
            z=Zs.tolist(),
            opacity=0.6,
            colorscale=color,
            name=name
        )
        data.append(surface.to_plotly_json())
    # Helper to create a cylinder surface
    def add_cylinder(expr, name, color):
        cyl_poly = sp.Poly(expr.expand(), x, y, z)
        const_term = cyl_poly.coeff_monomial((0, 0, 0))
        R = float(sp.sqrt(-float(const_term)))
        monoms = cyl_poly.monoms()
        sq_vars = set()
        for m in monoms:
            if m[0] == 2: sq_vars.add('x')
            if m[1] == 2: sq_vars.add('y')
            if m[2] == 2: sq_vars.add('z')
        axis_vars = {'x', 'y', 'z'} - sq_vars
        axis = axis_vars.pop() if axis_vars else 'z'
        theta_vals = np.linspace(0, 2*np.pi, 50)
        if axis == 'z':
            z_vals = np.linspace(bounds['z'][0], bounds['z'][1], 50)
            Theta, Zvals = np.meshgrid(theta_vals, z_vals)
            Xc = R * np.cos(Theta)
            Yc = R * np.sin(Theta)
            Zc = Zvals
        elif axis == 'x':
            x_vals = np.linspace(bounds['x'][0], bounds['x'][1], 50)
            Theta, Xvals = np.meshgrid(theta_vals, x_vals)
            Yc = R * np.cos(Theta)
            Zc = R * np.sin(Theta)
            Xc = Xvals
        else:  # axis == 'y'
            y_vals = np.linspace(bounds['y'][0], bounds['y'][1], 50)
            Theta, Yvals = np.meshgrid(theta_vals, y_vals)
            Xc = R * np.cos(Theta)
            Zc = R * np.sin(Theta)
            Yc = Yvals
        surface = go.Surface(
            x=Xc.tolist(),
            y=Yc.tolist(),
            z=Zc.tolist(),
            opacity=0.6,
            colorscale=color,
            name=name
        )
        data.append(surface.to_plotly_json())
    # Helper to create a paraboloid surface
    def add_paraboloid(expr, name, color):
        sol = sp.solve(expr, z)
        if sol:
            z_expr = sol[0]
            f_xy = sp.lambdify((x, y), z_expr, 'numpy')
            x_vals = np.linspace(bounds['x'][0], bounds['x'][1], 50)
            y_vals = np.linspace(bounds['y'][0], bounds['y'][1], 50)
            Xp, Yp = np.meshgrid(x_vals, y_vals)
            Zp = f_xy(Xp, Yp)
            surface = go.Surface(
                x=Xp.tolist(),
                y=Yp.tolist(),
                z=Zp.tolist(),
                opacity=0.6,
                colorscale=color,
                name=name
            )
            data.append(surface.to_plotly_json())
    # Helper to create a plane surface
    def add_plane(expr, name, color):
        plane_poly = sp.Poly(expr, x, y, z)
        if plane_poly.coeff_monomial((0, 0, 1)) != 0:
            sol = sp.solve(expr, z)
            if sol:
                z_expr = sol[0]
                f_xy = sp.lambdify((x, y), z_expr, 'numpy')
                x_vals = np.linspace(bounds['x'][0], bounds['x'][1], 50)
                y_vals = np.linspace(bounds['y'][0], bounds['y'][1], 50)
                Xp, Yp = np.meshgrid(x_vals, y_vals)
                Zp = f_xy(Xp, Yp)
                surface = go.Surface(
                    x=Xp.tolist(),
                    y=Yp.tolist(),
                    z=Zp.tolist(),
                    opacity=0.6,
                    colorscale=color,
                    name=name
                )
                data.append(surface.to_plotly_json())
        elif plane_poly.coeff_monomial((1, 0, 0)) != 0:
            sol = sp.solve(expr, x)
            if sol:
                x_expr = sol[0]
                f_yz = sp.lambdify((y, z), x_expr, 'numpy')
                y_vals = np.linspace(bounds['y'][0], bounds['y'][1], 50)
                z_vals = np.linspace(bounds['z'][0], bounds['z'][1], 50)
                Yp, Zp = np.meshgrid(y_vals, z_vals)
                Xp = f_yz(Yp, Zp)
                surface = go.Surface(
                    x=Xp.tolist(),
                    y=Yp.tolist(),
                    z=Zp.tolist(),
                    opacity=0.6,
                    colorscale=color,
                    name=name
                )
                data.append(surface.to_plotly_json())
        elif plane_poly.coeff_monomial((0, 1, 0)) != 0:
            sol = sp.solve(expr, y)
            if sol:
                y_expr = sol[0]
                f_xz = sp.lambdify((x, z), y_expr, 'numpy')
                x_vals = np.linspace(bounds['x'][0], bounds['x'][1], 50)
                z_vals = np.linspace(bounds['z'][0], bounds['z'][1], 50)
                Xp, Zp = np.meshgrid(x_vals, z_vals)
                Yp = f_xz(Xp, Zp)
                surface = go.Surface(
                    x=Xp.tolist(),
                    y=Yp.tolist(),
                    z=Zp.tolist(),
                    opacity=0.6,
                    colorscale=color,
                    name=name
                )
                data.append(surface.to_plotly_json())
        else:
            pass  # Unsupported plane orientation
    # Create surfaces for both expressions
    surfaces = [(expr1, type1, color1), (expr2, type2, color2)]
    for idx, (expr, t, color) in enumerate(surfaces, start=1):
        name = t.capitalize()
        if type1 == type2:
            name += f" {idx}"
        if t == 'sphere':
            add_sphere(expr, name, color)
        elif t == 'cylinder':
            add_cylinder(expr, name, color)
        elif t == 'paraboloid':
            add_paraboloid(expr, name, color)
        elif t == 'plane':
            add_plane(expr, name, color)
    layout = {
        'scene': {
            'xaxis_title': 'X',
            'yaxis_title': 'Y',
            'zaxis_title': 'Z'
        },
        'margin': {'l': 0, 'r': 0, 'b': 0, 't': 0},
        'showlegend': True
    }
    return {'data': data, 'layout': layout}

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        eq1 = request.form.get('equation1')
        eq2 = request.form.get('equation2')

        # Parse equations
        expr1, type1 = parse_equation(eq1)
        expr2, type2 = parse_equation(eq2)

        if None in [expr1, expr2, type1, type2]:
            return render_template('index.html', error="Invalid equations")

        # Get optimal coordinate system
        volume_coords = get_optimal_volume_coordinates(type1, type2)

        # Get optimal bounds
        bounds = get_optimal_bounds(expr1, type1, expr2, type2)

        # Calculate volume
        volume = estimate_volume(expr1, type1, expr2, type2, bounds)

        # Create 3D plot
        plot_data = create_3d_plot(expr1, type1, expr2, type2, bounds)

        result = {
            'coordinate_system': volume_coords,
            'volume': f"{volume:.4f}" if volume != float('inf') else "infinite"
        }

        return render_template('index.html', 
                             result=result,
                             plot_data=plot_data)
    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)
