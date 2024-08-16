# Object Detection on Google Street View Data

This Python module performs object detection on Google Street View photos fetched from specified geolocation boundaries. It allows downloading Google Street View images at evenly spaced points along the road network within the boundary. The module uses Grounding-SAM for detecting and segmenting objects based on text prompts from the images. Additionally, it estimates distances from image metadata geolocation using Zoe-Depth and includes custom functions to analyze the geospatial relationships between objects.

[slides](https://docs.google.com/presentation/d/1x6mj692BMaXJE4nDjVUDzIA_VW4Mc2wfDDEbevguVxM/edit?usp=sharing)  [Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/qlllllll/fire-risk-object-detection/blob/main/src/colab-demo.ipynb)
### Install

Install from git with:
```bash
git clone https://github.com/qlllllll/fire-risk-object-detection.git
pip install -r requirements.txt
```

### Quick Start

#### Downloading Google Street View Images of an Area

To download Google Street View images for a specific area, follow these steps:

1. Use `get_area_bbox` to access the Google Maps API and retrieve the bounding box for the specified area.
2. Use `generate_network_pts` to access the road network from OpenStreetMap using the geo boundary, and generate sample points along the network.
3. Use `load_gsv_img_from_coords` to load Google Street View images using the generated coordinates and save them in the specified directory.

```python
from geo_utils import get_area_bbox, generate_network_pts, load_gsv_img_from_coords

# Define the area of interest
area = 'Northside, Berkeley, CA'

# Retrieve the bounding box for the specified area
bbox = get_area_bbox(area)

# Generate sample points along the road network within the bounding box
sample_points = generate_network_pts(bbox, samp_dist=0.00015)
```

#### Object Detection on Series of Images

For object detection on a series of images loaded from a folder, the following functions operate on `pd.Series` for better readability and flow:

- `depth_estimate`: Returns a series of depth maps approximated with Zoe-Depth.
- `convert_depth_to_coords`: Projects pixels onto a 3D plane using a sample camera intrinsic matrix.
- `object_grounded_segmentation`: Returns a series of `DetectionResult` objects, which include [score, label, box, mask] from Grounding-SAM.
- `reformat_detections`: Returns a DataFrame of objects, organized by image and label.
- `generate_3d_bounding_boxes`: Estimates the corner coordinates of the 3D bounding box for the object series.

```python
from object_detection_utils import load_images, depth_estimate, convert_depth_to_coords, object_grounded_segmentation, generate_3d_bounding_boxes, reformat_detections

# Load the first 10 images from the coordinates into a pd.Series
images = load_gsv_img_from_coords(sample_points, api_key=GOOGLE_MAPS_API_KEY, save_dir='gsv_images')['image'][:10]
images.name = 'gsv'

# Estimate depth maps for the images
depth_maps = depth_estimate(images)

# Project pixels onto a 3D plane
coords = convert_depth_to_coords(depth_maps)

# Perform object detection
detections = object_grounded_segmentation(images, ['vegetation', 'house', 'fire hydrant'])

# Reformat detections into a DataFrame of objects
re_detections = reformat_detections(detections)

# Estimate 3D bounding boxes for objects
re_detections['coords'] = generate_3d_bounding_boxes(re_detections, coords)
re_detections.head(5)
```

#### Geospatial Relationships between Objects
To analyze the geospatial relationships between detected objects, use the following functions:

- `dist`: Calculates the minimum distances between two sets of bounding boxes.
- `group_distances`: Groups distances by object and applies a specified function to the minimum distances.

```python
from object_detection_utils import dist, group_distances

# Calculate distances between vegetation and house coordinates.
distances = dist(re_detections[['label','image_index', 'coords']], 'vegetation', 'house')

# Group the distances to vegetation by applying a minimum threshold and filter.
dist_house_to_veg_min = group_distances(distances, re_detections, 'vegetation', 'house', fn=min)
```

To check if the nearest objects exist within a specified distance, use the `nearest_object_existence` function. This function can also visualize the results if needed.

```python
from object_detection_utils import nearest_object_existence

# Check if fire hydrants exist within 0.001 distance from sample hydrants map
sample_hydrants = ...
nearest_exist = nearest_object_existence(sample_hydrants, re_detections[re_detections['label']=='fire hydrant'], meta=sample_points, max_dist=0.001, visualize=True)
```

To estimate the geographic locations of objects from bounding boxes, use the `estimate_object_locations` function. These functions can also support visualizing the estimated locations.

```python
from object_detection_utils import estimate_object_locations

# Estimate geographic locations for the detected objects
geoloc_results = estimate_object_locations(re_detections[['image_index','coords', 'label']], meta=sample_points, visualize=True)
```

For more information, see the [API Documentation](https://qlllllll.github.io/fire-risk-object-detection/).
