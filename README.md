# PerspectivePointCloud

This repo contains code for creating a point cloud that, when viewed from different angles, forms different images. It works by initializing a random point cloud and optimizing with PyTorch3D using a simple image difference loss function.

## Example

| Input 1 | Input 2 |
|  :---:  |  :---:  |
| ![Cat Drawing](/images/cat_face_line_drawing.jpeg?raw=true "Cat Drawing") | ![Dog Drawing](/images/dog_face.jpeg?raw=true "Dog Drawing") |
| Point Cloud Perspective 1  |  Point Cloud Perspective 2 |
| ![Cat Point Cloud](/images/cat_point_cloud.png?raw=true "Cat Point Cloud") | ![Dog Point Cloud](/images/dog_point_cloud.png?raw=true "Dog Point Cloud") | 

<h3 align="center">(It's the same point cloud)</h3>

## Get started

PyTorch3D requires `conda`. Install the environment `pytorch3d_notebooks.yml`.
