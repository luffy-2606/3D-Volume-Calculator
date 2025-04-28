# 3D-Volume-Calculator

## About
This project aims to develop an AI model to automate the selection of the optimal coordinate system (Cartesian, Cylindrical, or Spherical) for solving triple integrals involving two surfaces. By leveraging machine learning and symbolic computation, the system simplifies volume calculations in multivariable calculus, providing faster and more accurate solutions.

## Introduction
Triple integration is essential for computing volumes bounded by surfaces in 3D space. Traditionally, selecting the optimal coordinate system relies on human intuition. This project trains an AI model to predict the best coordinate system based on surface equations and subsequently computes the volume using triple integration.

## Objectives
- Develop a dataset of surface equations paired with optimal coordinate systems.
- Train a machine learning model (Random Forest Classifier) to predict the optimal coordinate system.
- Implement a system to calculate volume integrals after coordinate selection.
- Validate the model’s accuracy in coordinate system selection and integration correctness.

## Technologies Used
| Technology       | Purpose                          |
|------------------|----------------------------------|
| **Flask**        | Web framework for the front-end  |
| **SymPy**        | Symbolic math parsing/manipulation |
| **NumPy**        | Numerical computations           |
| **Plotly**       | 3D visualization of regions      |
| **Scikit-learn** | Random Forest Classifier for ML  |

## System Components

### Equation Parsing
- **Function**: `parse_equation(equation_str: str)`
- Converts user input into SymPy expressions and identifies region types (e.g., sphere, cylinder, paraboloid).

### Coordinate System Identification
- **Function**: `find_coordinate_system(equation_str: str)`
- Uses a trained Random Forest Classifier to predict the optimal coordinate system based on features like \(x^2\), \(y^2\), and \(z\).

### Bounds Calculation
- **Function**: `get_optimal_bounds(expr1, type1, expr2, type2)`
- Dynamically computes bounding boxes for Monte Carlo point generation based on region types.

### Monte Carlo Volume Estimation
- Generates random points within bounds and estimates the enclosed volume by checking inclusion in both regions.

### Optimal Coordinate Recommendation
- Recommends the most efficient coordinate system for analytical integration if applicable.

### Flask Web Server
- Provides a user interface to input equations, processes data, and returns results (volume and coordinate system).

## Results
**Sample Test Case**  
- **Region 1**: Sphere \(x^2 + y^2 + z^2 = 4\) (radius 2)  
- **Region 2**: Plane \(z = 0\) (upper half-space)  
- **Predicted Coordinate System**: Spherical  
- **Estimated Volume**: 16.7552 cubic units  

## Conclusion
The project successfully demonstrates the feasibility of automating coordinate system selection and triple integral computation using AI. Challenges included handling edge cases with ambiguous coordinate systems and complex boundary conditions. Future enhancements could expand to more coordinate systems, improve numerical methods, and integrate advanced symbolic solvers.

---

**Contributors**: Azan Wasty, Ali Naveed
