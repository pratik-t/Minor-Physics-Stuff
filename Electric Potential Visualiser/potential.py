# import numpy as np
# import plotly.graph_objects as go

# # Define the function V(x, y, z)
# def V(n, x, y, z, a, b, c):

#     # Phi = 0
#     # for i in range(1, n, 2):
#     #     for j in range(1, n, 2):
#     #         Phi += np.exp(-np.pi*np.sqrt(i**2+j**2)*x)*np.sin(i*np.pi*y)*np.sin(j*np.pi*z)/(i*j)
    
#     # Phi *= 16/(np.pi**2)

    
#     r = np.sqrt(x**2+y**2+z**2)
#     ct = z/r
#     R = 1
#     a = 3

#     Phi = 1/np.sqrt(r**2+a**2-2*r*a*ct) - 1/np.sqrt(R**2+(r*a/R)**2-2*r*a*ct)

#     return (Phi)

# # Generate a grid of points in 3D space
# n = 100
# a = 10
# b = 10
# c = 10

# x = np.linspace(-10, 10, n)
# y = np.linspace(-10, 10, n)
# z = np.linspace(-10, 10, n)

# X, Y, Z = np.meshgrid(x, y, z)

# # Compute the function V on the grid
# V_values = V(4, X, Y, Z, a, b, c)

# # Create a volume rendering of the scalar field V
# fig = go.Figure(data=go.Volume(
#     x=X.flatten(),
#     y=Y.flatten(),
#     z=Z.flatten(),
#     value=V_values.flatten(),
#     isomin=-1,
#     isomax=1,
#     opacity=0.1,  # Adjust the opacity to visualize the density
#     surface_count=30,  # Number of layers (isosurfaces) to draw
#     colorscale='Viridis'
# ))

# # Set plot title and layout
# fig.update_layout(scene=dict(aspectmode='cube'), title="3D Density Plot of V(x, y, z)")
# fig.show()






import numpy as np
import plotly.graph_objects as go

# Define the 2D function f(x, y)
def f(x, y):
    return -(2/(2*np.pi*(x**2+y**2+4)**(3/2)))

# Generate a grid of points in 2D space
x = np.linspace(-50, 50, 100)
y = np.linspace(-50, 50, 100)
X, Y = np.meshgrid(x, y)

Z = f(X, Y)

# Create a surface plot
fig = go.Figure(data=[go.Surface(z=Z, x=X, y=Y, colorscale='Viridis')])

fig.update_layout(title="Surface Plot of f(x, y)", scene=dict(
    xaxis_title='X',
    yaxis_title='Y',
    zaxis_title='f(x, y)'
))

fig.show(renderer='browser')
