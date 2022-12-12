# PerspectivePointCloud

This repo contains code for creating a point cloud that, when viewed from different angles, forms different images. It works by initializing a random point cloud and optimizing with PyTorch3D using a simple image difference loss function.

## Example

| Input 1 | Input 2 | Input 2 |
|  :---:  |  :---:  |  :---:  |
| ![Cat Drawing](/images/cat_face_line_drawing.jpeg?raw=true "Cat Drawing") | ![Dog Drawing](/images/dog_face.jpeg?raw=true "Dog Drawing") | ![Guitar Drawing](/images/guitar_line_drawing.jpeg?raw=true "Guitar Drawing")
| Point Cloud Perspective 1  |  Point Cloud Perspective 2 |  Point Cloud Perspective 3 |
| ![Cat Point Cloud](/images/cat_point_cloud.png?raw=true "Cat Point Cloud") | ![Dog Point Cloud](/images/dog_point_cloud.png?raw=true "Dog Point Cloud") | ![GuitarOutput](/images/guitar_point_cloud.png?raw=true "Guitar Point Cloud") |

<h3 align="center">(It's the same point cloud)</h3>



https://user-images.githubusercontent.com/47218308/206943774-d34c71bd-a524-40fd-917b-6d3cbae778e2.mp4



## Get started

PyTorch3D requires `conda`. Install the environment `pytorch3d_notebooks.yml`.
