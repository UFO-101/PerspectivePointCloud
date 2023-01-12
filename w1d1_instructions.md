
# W1D1: Triangle Rendering

Today we'll be practicing batched matrix operations in PyTorch by writing a basic graphics renderer. We'll start with an extremely simplified case and work up to rendering your very own 3D Pikachu! Note that if you're viewing this file on GitHub, some of the equations may not render properly. Viewing it locally in VS Code should fix this.

## Table of Contents

- [Readings](#readings)
- [1D Image Rendering](#d-image-rendering)
    - [Tip - the `out` keyword argument](#tip---the-out-keyword-argument)
- [Ray-Object Intersection](#ray-object-intersection)
- [Batched Ray-Segment Intersection](#batched-ray-segment-intersection)
    - [Tip - Ellipsis](#tip---ellipsis)
- [Tip - Elementwise Logical Operations on Tensors](#tip---elementwise-logical-operations-on-tensors)
    - [Tip - Operator Precedence](#tip---operator-precedence)
    - [Tip - Logical Reductions](#tip---logical-reductions)
- [2D Rays](#d-rays)
- [Triangle Coordinates](#triangle-coordinates)
    - [Triangle-Ray Intersection](#triangle-ray-intersection)
- [Single-Triangle Rendering](#single-triangle-rendering)
    - [Views and Copies](#views-and-copies)
    - [Storage Objects](#storage-objects)
    - [`Tensor._base`](#tensorbase)
- [Mesh Loading](#mesh-loading)
- [Mesh Rendering](#mesh-rendering)
- [Bonus Content](#bonus-content)

## Readings

None!

## 1D Image Rendering

In our initial setup, the **camera** will be a single point at the origin, and the **screen** will be the plane at x=1.

**Objects** in the world consist of triangles, where triangles are represented as 3 points in 3D space (so 9 floating point values per triangle). You can build any shape out of sufficiently many triangles and your Pikachu will be made from 412 triangles.

The camera will emit one or more **rays**, where a ray is represented by an **origin** point and a **direction** point. Conceptually, the ray is emitted from the origin and continues in the given direction until it intersects an object.

We have no concept of lighting or color yet, so for now we'll say that a pixel on our screen should show a bright color if a ray from the origin through it intersects an object, otherwise our screen should be dark.

<p align="center">
    <img src="w1d1_ray_tracing.png"/>
</p>

To start, we'll let the z dimension in our `(x, y, z)` space be zero and work in the remaining two dimensions.

Implement the following `make_rays_1d` function so it generates some rays coming out of the origin, which we'll take to be `(0, 0, 0)`.

Calling `render_lines_with_pyplot` on your rays should look like this (note the orientation of the axes):

<p align="center">
    <img src="w1d1_make_rays_1d.png" width="400" />
</p>


```python
import os
import torch as t
import einops
import matplotlib.pyplot as plt
from ipywidgets import interact
import w1d1_test

MAIN = __name__ == "__main__"
IS_CI = os.getenv("IS_CI")


def make_rays_1d(num_pixels: int, y_limit: float) -> t.Tensor:
    """
    num_pixels: The number of pixels in the y dimension. Since there is one ray per pixel, this is also the number of rays.
    y_limit: At x=1, the rays should extend from -y_limit to +y_limit, inclusive of both endpoints.

    Returns: shape (num_pixels, num_points=2, num_dim=3) where the num_points dimension contains (origin, direction) and the num_dim dimension contains xyz.

    Example of make_rays_1d(9, 1.0): [
        [[0, 0, 0], [1, -1.0, 0]],
        [[0, 0, 0], [1, -0.75, 0]],
        [[0, 0, 0], [1, -0.5, 0]],
        ...
        [[0, 0, 0], [1, 0.75, 0]],
        [[0, 0, 0], [1, 1, 0]],
    ]
    """
    pass


def render_lines_with_pyplot(lines: t.Tensor):
    """Plot any number of line segments in 3D.

    lines: shape (num_lines, num_points=2, num_dims=3).
    """
    (fig, ax) = plt.subplots(figsize=(10, 10), subplot_kw={"projection": "3d"})
    for line in lines:
        ax.plot(line[:, 1].numpy(), line[:, 0].numpy(), line[:, 2].numpy())
    ax.set(xlabel="Y", ylabel="X", zlabel="Z")
    return fig


rays1d = make_rays_1d(9, 10.0)
if MAIN and (not IS_CI):
    render_lines_with_pyplot(rays1d)

```

### Tip - the `out` keyword argument

Many PyTorch functions take an optional keyword argument `out`. If provided, instead of allocating a new tensor and returning that, the output is written directly to the `out` tensor.

If you used `torch.arange` or `torch.linspace` above, try using the `out` argument. Note that a basic indexing expression like `rays[:, 1, 1]` returns a view that shares storage with `rays`, so writing to the view will modify `rays`. You'll learn more about views later today.

## Ray-Object Intersection

Suppose we have a line segment defined by points $L_1$ and $L_2$. Then for a given ray, we can test if the ray intersects the line segment like so:

- Supposing both the ray and line segment were infinitely long, solve for their intersection point.
- If the point exists, check whether that point is inside the line segment and the ray.

Our camera ray is defined by the origin $O$ and direction $D$ and our object line is defined by points $L_1$ and $L_2$.

We can write the equations for all points on the camera ray as $R(u)=O +u D$ for $u \in [0, \infty)$ and on the object line as $O(v)=L_1+v(L_2 - L_1)$ for $v \in [0, 1]$.

The following interactive widget lets you play with this parameterization of the problem:


```python
@interact
def line(v=(-2.0, 2.0), seed=(0, 10)):
    """
    Interactive line widget.

    Drag "seed" to get a different random line.
    Drag "v" to see that v must be in [0, 1] for the intersection marked by a star to be "inside" the object line.
    """
    t.manual_seed(seed)
    L_1 = t.randn(2)
    L_2 = t.randn(2)
    P = lambda v: L_1 + v * (L_2 - L_1)
    (x, y) = zip(P(-2), P(2))
    plt.plot(x, y, "g-")
    plt.plot(*L_1, "ro", markersize=12)
    plt.plot(*L_2, "ro", markersize=12)
    plt.plot(P(v)[0], P(v)[1], "*", markersize=12)
    plt.xlabel("X")
    plt.ylabel("Y")

```

Setting the line equations from above equal gives the solution:

$$\begin{aligned}O + u D &= L_1 + v(L_2 - L_1) \\ u D - v(L_2 - L_1) &= L_1 - O  \\ \begin{pmatrix} D_x & (L_1 - L_2)_x \\ D_y & (L_1 - L_2)_y \\ \end{pmatrix} \begin{pmatrix} u \\ v \\ \end{pmatrix} &=  \begin{pmatrix} (L_1 - O)_x \\ (L_1 - O)_y \\ \end{pmatrix} \end{aligned}$$

Once we've found values of $u$ and $v$ which satisfy this equation, if any (the lines could be parallel) we just need to check that $u \geq 0$ and $v \in [0, 1]$.

Exercise: for each of the following segments, which camera rays from earlier intersect? You can do this by inspection or using `render_lines_with_pyplot`.

<details>

<summary>Solution - Intersecting Rays</summary>

- Segment 0 intersects the first two rays.
- Segment 1 doesn't intersect any rays.
- Segment 2 intersects the last two rays. Computing `rays * 2` projects the rays out to `x=1.5`. Remember that while the plot shows rays as line segments, rays conceptually extend indefinitely.

</details>


```python
segments = t.tensor(
    [[[1.0, -12.0, 0.0], [1, -6.0, 0.0]], [[0.5, 0.1, 0.0], [0.5, 1.15, 0.0]], [[2, 12.0, 0.0], [2, 21.0, 0.0]]]
)
"TODO: YOUR CODE HERE"

```

Using [`torch.lingalg.solve`](https://pytorch.org/docs/stable/generated/torch.linalg.solve.html) and [`torch.stack`](https://pytorch.org/docs/stable/generated/torch.stack.html), implement the `intersect_ray_1d` function to solve the above matrix equation.

Is it possible for the solve method to fail? Give a sample input where this would happen.

<details>

<summary>Solution - Failing Solve</summary>

If the ray and segment are exactly parallel, then the solve will fail because there is no solution to the system of equations. For this function, handle this by catching the exception and returning False.

</details>

<details>

<summary>Help! My code is failing with a "must be batches of square matrices" exception.</summary>

Our formula only uses the x and y coordinates - remember to discard the z coordinate for now. It's good practice to write asserts on the shape of things so that your asserts will fail with a helpful error message. In this case, you could assert that the `A` argument is of shape (2, 2) and the `B` argument is of shape (2,)

</details>


```python
def intersect_ray_1d(ray: t.Tensor, segment: t.Tensor) -> bool:
    """
    ray: shape (n_points=2, n_dim=3)  # O, D points
    segment: shape (n_points=2, n_dim=3)  # L_1, L_2 points

    Return True if the ray intersects the segment.
    """
    pass


w1d1_test.test_intersect_ray_1d(intersect_ray_1d)
w1d1_test.test_intersect_ray_1d_special_case(intersect_ray_1d)

```

## Batched Ray-Segment Intersection

Next, implement a batched version that takes multiple rays, multiple line segments, and returns a boolean for each ray indicating whether **any** segment intersects with that ray.

In the batched version, we don't want the solver to throw an exception just because some of the equations don't have a solution - these should just return False. What's one way to achieve this, ignoring efficiency for now?

<details>

<summary>Solution - Detecting No Solution</summary>

The error appears when the matrix is not invertible. We call this a singular matrix, and it's equivalent to a determinant of zero.

`torch.linalg.det` can compute the determinant, and if it's close to zero (allowing for floating point error) then we know there's no solution. One approach would be filtering out those batch elements using a mask, but since this rarely happens it is convenient and not very expensive to just replace that element with something we know is invertible like the identity matrix, and then overwrite the output in that position with False.

</details>

### Tip - Ellipsis

You can use an ellipsis `...` in an indexing expression to avoid repeated `:' and to write indexing expressions that work on varying numbers of input dimensions.

For example, `x[..., 0]` is equivalent to `x[:, :, 0]` if `x` is 3D, and equivalent to `x[:, :, :, 0]` if `x` is 4D.

## Tip - Elementwise Logical Operations on Tensors

For regular booleans, the keywords `and`, `or`, and `not` are used to do logical operations and the operators `&`, `|`, and `~` do and, or and not on each bit of the input numbers. For example `0b10001 | 0b11000` is `0b11001` or 25 in base 10.

Tragically, Python doesn't allow classes to overload keywords, so if `x` and `y` are of type `torch.Tensor`, then `x and y` does **not** do the natural thing that you probably expect, which is compute `x[i] and y[i]` elementwise. It actually tries to coerce `x` to a regular boolean, which throws an exception.

As a workaround, PyTorch (and NumPy) have chosen to overload the bitwise operators but have them actually mean logical operations, since you usually don't care to do bitwise operations on tensors. So the correct expression would be `x & y` to compute `x[i] and y[i]` elementwise.

### Tip - Operator Precedence

Another thing that tragically doesn't do what you would expect is an expression like `v >= 0 & v <= 1`. The operator precedence of `&` is so high that this statement parses as `(v >= (0 & v)) <= 1`.

The correct expression uses parentheses to force the proper parsing: `(v >= 0) & (v <= 1)`.

### Tip - Logical Reductions

In plain Python, if you have a list of lists and want to know if any element in a row is `True`, you could use a list comprehension like `[any(row) for row in rows]`. The efficient way to do this in PyTorch is with `torch.any()` or equivalently the `.any()` method of a tensor, which accept the dimension to reduce over. Similarly, `torch.all()` or `.all()` method.

You can accomplish the same thing with `einops.reduce` but that's more cumbersome.


```python
def intersect_rays_1d(rays: t.Tensor, segments: t.Tensor) -> t.Tensor:
    """
    rays: shape (NR, 2, 3) - NR is the number of rays
    segments: shape (NS, 2, 3) - NS is the number of segments

    Return: shape (NR, )
    """
    pass


if MAIN:
    w1d1_test.test_intersect_rays_1d(intersect_rays_1d)
    w1d1_test.test_intersect_rays_1d_special_case(intersect_rays_1d)

```

## 2D Rays

Now we're going to make use of the z dimension and have rays emitted from the origin in both y and z dimensions.

Implement `make_rays_2d` analogously to `make_rays_1d`. The result should look like a pyramid with the tip at the origin.

<details>

<summary>Spoiler - Help with make_rays_2d</summary>

Don't write it as a function right away. The most efficient way is to write and test each line individually in the REPL to verify it does what you expect before proceeding.

You can either build up the output tensor using `torch.stack`, or you can initialize the output tensor to its final size and then assign to slices like `rays[:, 1, 1] = ...`. It's good practice to be able to do it both ways.

Each y coordinate needs a ray with each corresponding z coordinate - in other words this is an outer product. The most elegant way to do this is with two calls to `einops.repeat`. You can also accomplish this with `unsqueeze`, `expand`, and `reshape` combined.

</details>


```python
def make_rays_2d(num_pixels_y: int, num_pixels_z, y_limit: float, z_limit: float) -> t.Tensor:
    """
    num_pixels_y: The number of pixels in the y dimension
    num_pixels_z: The number of pixels in the z dimension

    y_limit: At x=1, the rays should extend from -y_limit to +y_limit, inclusive of both.
    z_limit: At x=1, the rays should extend from -z_limit to +z_limit, inclusive of both.

    Returns: shape (num_rays=num_pixels_y * num_pixels_z, num_points=2, num_dims=3).
    """
    pass


if MAIN and (not IS_CI):
    rays_2d = make_rays_2d(10, 10, 0.3, 0.3)
    render_lines_with_pyplot(rays_2d)

```

## Triangle Coordinates

The area inside a triangle can be defined by three (non-collinear) points $A$, $B$ and $C$, and can be written algebraically as:

$$P(w, u, v) = wA + uB + vC$$
$$s.t.$$
$$0 \leq w,u,v$$
$$w + u + v = 1$$

Or equivalently:

$$P(u, v) = (1 - u - v)A + uB + vC =$$
$$P(u, v) = A + u(B - A) + v(C - A)$$
$$s.t.$$
$$0 \leq u,v$$
$$u + v \leq 1$$

These $u, v$ are called "barycentric coordinates".

If we remove the bounds on $u$ and $v$, we get an equation for the plane containing the triangle. Play with the widget to understand the behavior of $u, v$.


```python
one_triangle = t.tensor([[0, 0, 0], [3, 0.5, 0], [2, 3, 0]])


@interact
def triangle(u=(-1.0, 2.0), v=(-1.0, 2.0)):
    (A, B, C) = one_triangle
    for p in (A, B, C):
        plt.plot(p[0], p[1], "ro", markersize=12)
    for (p, q) in ((A, B), (B, C), (C, A)):
        (x, y, _) = zip(p, q)
        plt.plot(x, y, "g-")
    P = A + u * (B - A) + v * (C - A)
    plt.plot(P[0], P[1], "*", markersize=12)
    plt.xlabel("X")
    plt.ylabel("Y")

```

### Triangle-Ray Intersection

Given a ray with origin $O$ and direction $D$, our intersection algorithm will consist of two steps:

- Finding the intersection between the line and the plane containing the triangle, by solving the equation $P(s) = P(u, v)$;
- Checking if $u$ and $v$ are within the bounds of the triangle.

Expanding the equation $P(s) = P(u, v)$, we have:

$$O + sD = A + u(B - A) + v(C - A) \Rightarrow$$

$$\begin{gather*} \begin{pmatrix} -D & (B - A) & (C - A) \\ \end{pmatrix}  \begin{pmatrix} s \\ u \\ v  \end{pmatrix} = \begin{pmatrix} (O - A) \end{pmatrix} \Rightarrow \end{gather*} \newline \begin{gather*} \begin{pmatrix} -D_x & (B - A)_x & (C - A)_x \\
-D_y & (B - A)_y & (C - A)_y \\ -D_z & (B - A)_z & (C - A)_z \\ \end{pmatrix}  \begin{pmatrix}s \\ u \\ v  \end{pmatrix} = \begin{pmatrix}  (O - A)_x \\ (O - A)_y \\ (O - A)_z \\ \end{pmatrix} \end{gather*}$$

We can therefore find the coordinates `s`, `u`, `v` of the intersection point by solving the linear system above.

Using `torch.linalg.solve` and `torch.stack`, implement `triangle_line_intersects(A, B, C, O, D)`.

Tip: if you have a 0-dimensional tensor with shape `()` containing a single value, use the `item()` method to convert it to a plain Python value.

Tip: if it's not working, try making a simple ray and triangle with nice round numbers where you can work out manually if it should intersect or not, then debug from there.


```python
def triangle_line_intersects(A: t.Tensor, B: t.Tensor, C: t.Tensor, O: t.Tensor, D: t.Tensor) -> bool:
    """
    A: shape (3,), one vertex of the triangle
    B: shape (3,), second vertex of the triangle
    C: shape (3,), third vertex of the triangle
    O: shape (3,), origin point
    D: shape (3,), direction point

    Return True if the line and the triangle intersect.
    """
    pass


if MAIN:
    w1d1_test.test_triangle_line_intersects(triangle_line_intersects)

```

## Single-Triangle Rendering

Implement `raytrace_triangle` using only one call to `torch.linalg.solve`.

Reshape the output and visualize with `plt.imshow`. It's normal for the edges to look pixelated and jagged - using a small number of pixels is a good way to debug quickly.

If you think it's working, increase the number of pixels and verify that it looks less pixelated at higher resolution.

### Views and Copies

It's critical to know when you are making a copy of a `Tensor`, versus making a view of it that shares the data with the original tensor. It's preferable to use a view whenever possible to avoid copying memory unnecessarily. On the other hand, modifying a view modifies the original tensor which can be unintended and surprising. Consult [the documentation](https://pytorch.org/docs/stable/tensor_view.html) if you're unsure if a function returns a view. A short reference of common functions:

- `torch.expand`: always returns a view
- `torch.view`: always returns a view
- `torch.detach`: always returns a view
- `torch.repeat`: always copies
- `torch.clone`: always copies
- `torch.flip`: always copies (different than numpy.flip which returns a view)
- `torch.tensor`: always copies, but PyTorch recommends using `.clone().detach()` instead.
- `torch.Tensor.contiguous`: returns self if possible, otherwise a copy
- `torch.transpose`: returns a view if possible, otherwise (sparse tensor) a copy
- `torch.reshape`: returns a view if possible, otherwise a copy
- `torch.flatten`: returns a view if possible, otherwise a copy (different than numpy.flatten which returns a copy)
- `einops.repeat`: returns a view if possible, otherwise a copy
- `einops.rearrange`: returns a view if possible, otherwise a copy
- Basic indexing returns a view, while advanced indexing returns a copy.

### Storage Objects

Calling `storage()` on a `Tensor` returns a Python object wrapping the underlying C++ array. This array is 1D regardless of the dimensionality of the `Tensor`. This allows you to look inside the `Tensor` abstraction and see how the actual data is laid out in RAM.

Note that a new Python wrapper object is generated each time you call `storage()`, and both `x.storage() == x.storage()` and `x.storage() is x.storage()` evaluates to False.

If you want to check if two `Tensor`s share an underlying C++ array, you can compare their `storage().data_ptr()` fields. This can be useful for debugging.

### `Tensor._base`

If `x` is a view, you can access the original `Tensor` with `x._base`. This is an undocumented internal feature that's useful to know. Consider the following code:

```python
x = t.zeros(1024*1024*1024)
y = x[0]
del x
```

Here, `y` was created through basic indexing, so `y` is a view and `y._base` refers to `x`. This means `del x` won't actually deallocate the 4GB of memory, and that memory will remain in use which can be quite surprising. `y = x[0].clone()` would be an alternative here that does allow reclaiming the memory.



```python
def raytrace_triangle(triangle: t.Tensor, rays: t.Tensor) -> t.Tensor:
    """For each ray, return True if the triangle intersects that ray.

    triangle: shape (n_points=3, n_dims=3)
    rays: shape (n_pixels, n_points=2, n_dims=3)

    return: shape (n_pixels, )
    """
    pass


if MAIN and (not IS_CI):
    A = t.tensor([2, 0.0, -1.0])
    B = t.tensor([2, -1.0, 0.0])
    C = t.tensor([2, 1.0, 1.0])
    num_pixels_y = num_pixels_z = 10
    y_limit = z_limit = 0.5
    "TODO: YOUR CODE HERE"

```

## Mesh Loading

Use the given code to load the triangles for your Pikachu. By convention, files written with `torch.save` end in the `.pt` extension, but these are actually just zip files.


```python
with open("w1d1_pikachu.pt", "rb") as f:
    triangles = t.load(f)

```

## Mesh Rendering

For our purposes, a mesh is just a group of triangles, so to render it we'll intersect all rays and all triangles at once. We previously just returned a boolean for whether a given ray intersects the triangle, but now it's possible that more than one triangle intersects a given ray.

For each ray (pixel) we will return a float representing the minimum distance to a triangle if applicable, otherwise the special value `float('inf')` representing infinity. We won't return which triangle was intersected for now.

Implement `raytrace_mesh` and as before, reshape and visualize the output. Your Pikachu is centered on (0, 0, 0), so you'll want to slide the ray origin back to at least `x=-2` to see it properly.

Tip: use the `amin()` method of a tensor to find the minimum along a dimension.

Tip: you can manually plot the triangles using lines to help visualize where they are relative to your rays.


```python
def raytrace_mesh(triangles: t.Tensor, rays: t.Tensor) -> t.Tensor:
    """For each ray, return the distance to the closest intersecting triangle, or infinity.

    triangles: shape (n_triangles, n_points=3, n_dims=3)
    rays: shape (n_pixels, n_points=2, n_dims=3)

    return: shape (n_pixels, )
    """
    pass


num_pixels_y = 120
num_pixels_z = 120
y_limit = z_limit = 1
"TODO: YOUR CODE HERE"

```

## Bonus Content

Congratulations, you've finished the main content for today!

Some fun extensions to try:

- Vectorize further to make a video.
    - Each frame will have its own rays coming from a slightly different position.
    - Pan the camera around for some dramatic footage.
    - One way to do it is using the `mediapy` library to render the video.
- Try rendering on the GPU and see if you can make it faster.
- Allow each triangle to have a corresponding RGB color value and render a colored image.
- Use multiple rays per pixel and combine them somehow to have smoother edges.