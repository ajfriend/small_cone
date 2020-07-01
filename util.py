import cvxpy as cvx

import numpy as np
from numpy import sin, cos, pi

import matplotlib.pyplot as plt
import plotly.graph_objects as go


def foo(x):
    return cvx.norm(x[:d]) <= x[d]

def perp(b):
    A = np.hstack([b[:, np.newaxis], np.eye(3)])

    q,r = np.linalg.qr(A)

    return q[:,1:].T

def sphere_mesh():
    # make a plotly 3d mesh of a sphere
    phi = np.linspace(0, 2*pi)
    theta = np.linspace(-pi/2, pi/2)
    phi, theta=np.meshgrid(phi, theta)

    x = cos(theta) * sin(phi) 
    y = cos(theta) * cos(phi)
    z = sin(theta)

    mesh = go.Mesh3d({
                    'x': x.flatten(), 
                    'y': y.flatten(), 
                    'z': z.flatten(), 
                    'alphahull': 0,
                    'opacity' :0.10,
    })
    
    return mesh

def sphere_scatter(*clouds):
    
    data = [ sphere_mesh() ] + [
        go.Scatter3d(
            x=c[0],
            y=c[1],
            z=c[2],
            mode='markers',
            opacity= 1 if i==0 else .2
        )
        for i,c in enumerate(clouds)
    ]
    
    fig = go.Figure(data=data)
    fig.update_layout(scene_aspectmode='cube')
    fig.show()
    
    
def small_cone(x, verbose=False):
    """
    Parameters
    ----------
    x : numpy.array
        An N x d array, where d is the dimension and N is the number of points
    """
    N, d = x.shape

    A = cvx.Variable((d,d), PSD=True)
    b = cvx.Variable(d)

    cst = [
        cvx.norm(A@x[i]) <= b@x[i]
        for i in range(N)
    ] + [
        cvx.matrix_frac(b, A) <= 1
    ]

    obj = cvx.Minimize(-cvx.log_det(A))
    prob = cvx.Problem(obj, cst)
    prob.solve(verbose=verbose, eps=1e-8)

    A = A.value
    b = b.value
    
    return A, b, prob

def small_cone2(x, verbose=False):
    """
    Parameters
    ----------
    x : numpy.array
        An N x d array, where d is the dimension and N is the number of points
    """
    N, d = x.shape
    
    x = x/np.linalg.norm(x, axis=1)[:, np.newaxis]

    A = cvx.Variable((d,d), PSD=True)

    cst = [
        cvx.norm(A@x[i]) <= 1
        for i in range(N)
    ]

    obj = cvx.Minimize(-cvx.log_det(A))
    prob = cvx.Problem(obj, cst)
    prob.solve(verbose=verbose, eps=1e-8)

    A = A.value
    
    return A, prob


def matrix_frac(b, A):
    return cvx.matrix_frac(b, A).value


def show_ellipse(points, line):
    
    data = [ sphere_mesh() ] + [
        go.Scatter3d(
            x=points[0],
            y=points[1],
            z=points[2],
            mode='markers',
            opacity= 1
        )
    ] + [
        go.Scatter3d(
            x = line[0],
            y = line[1],
            z = line[2],
            line=dict(
                color='darkblue',
                width=2
            ),
            marker=dict(
                size=0,
#                 color=z,
#                 colorscale='Viridis',
            ),
        )
    ]
    
    fig = go.Figure(data=data)
    fig.update_layout(scene_aspectmode='cube')
    fig.show()
    
    
def make_circle(A,b):
    c = np.linalg.solve(A,b)
    q = perp(c).T
    
    phi = np.linspace(0, 2*pi, 1000)

    u = np.array([cos(phi), sin(phi)])
    u = q@u
    
    c2 = np.linalg.norm(c)**2
    c = c/c2

    u = c + np.sqrt(1 - 1/c2)*u.T
    u = np.linalg.solve(A,u.T)
    u = u/np.linalg.norm(u,axis=0)
    
    return u