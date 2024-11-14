import numpy as np
import cv2
import pandas as pd
import matplotlib.pyplot as plt
import random
from typing import Union, List, Optional, Dict
from PIL import Image
import webcolors

from object_detection_utils import *

def annotate(image: Union[Image.Image, np.ndarray], detection_results: List[DetectionResult]) -> np.ndarray:
    """
    Annotate image with bounding boxes and masks for detection results.

    Args:
    - image (Union[Image.Image, np.ndarray]): The image to be annotated.
    - detection_results (List[DetectionResult]): List of detection results.

    Returns:
    - np.ndarray: Annotated image in numpy array format.
    """
    # Convert PIL Image to OpenCV format
    image_cv2 = np.array(image) if isinstance(image, Image.Image) else image
    #image_cv2 = cv2.cvtColor(image_cv2, cv2.COLOR_RGB2BGR)

    # Iterate over detections and add bounding boxes and masks
    for detection in detection_results:
        label = detection.label
        score = detection.score
        box = detection.box
        mask = detection.mask

        # Sample a random color for each detection
        color = np.random.randint(0, 256, size=3)

        # Draw bounding box
        cv2.rectangle(image_cv2, (box.xmin, box.ymin), (box.xmax, box.ymax), color.tolist(), 2)
        cv2.putText(image_cv2, f'{label}: {score:.2f}', (box.xmin, box.ymin - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color.tolist(), 2)

        # If mask is available, apply it
        if mask is not None:
            # Convert mask to uint8
            mask_uint8 = (mask * 255).astype(np.uint8)
            contours, _ = cv2.findContours(mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(image_cv2, contours, -1, color.tolist(), 2)

    return cv2.cvtColor(image_cv2, cv2.COLOR_BGR2RGB)

def annotate_mask_dist(image_df: pd.DataFrame, dist_df: pd.DataFrame, label1: str, label2: str, img_idx: int, obj_idx: int, save_name: Optional[str] = None):
    """
    Annotate image with masks and distance information for specified labels.

    Args:
    - image (Union[Image.Image, np.ndarray]): The image to be annotated.
    - row (pd.Series): The row containing the data for annotation.
    - label1 (str): First label for annotation.
    - label2 (str): Second label for annotation.
    - save_name (Optional[str]): Path to save the annotated image. Defaults to None.

    Returns:
    - None
    """
    plt.figure(figsize=(10, 6))
    ax = plt.gca()
    image = image_df.iloc[img_idx].image
    row = dist_df[(dist_df[f'obj_idx_{label1}'] == obj_idx) & (dist_df[f'img_idx'] == img_idx)]
    image_cv2 = np.array(image) if isinstance(image, Image.Image) else image
    #image_cv2 = cv2.cvtColor(image_cv2, cv2.COLOR_RGB2BGR)

    padding = 15
    image_cv2 = cv2.copyMakeBorder(image_cv2, padding, padding, padding, padding, cv2.BORDER_CONSTANT, value=[255, 255, 255])

    colors = random_named_css_colors(2)

    for index, lbl in enumerate([label1, label2]): 
        xmin, ymin, _, _ = row[f'box_{lbl}'].iloc[0]
        xmin += padding
        ymin += padding
    
        mask = np.array(row[f'mask_{lbl}'].iloc[0])
        mask_uint8 = (mask * 255).astype(np.uint8)
        contours, _ = cv2.findContours(mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        color_name = colors[index]
        color = webcolors.name_to_rgb(color_name)
        color = (color.red, color.green, color.blue)
        cv2.drawContours(image_cv2, [c + [padding, padding] for c in contours], -1, color, 2)

        obj_idx = row[f"obj_idx_{lbl}"].iloc[0]
        text = f'{lbl} [{obj_idx}]: {row["distance"].iloc[0]:.5f}'
        (text_width, text_height), baseline = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(image_cv2, (xmin + 5, ymin - text_height - baseline), 
                      (xmin + 5 + text_width, ymin + baseline), color, thickness=cv2.FILLED)
        cv2.putText(image_cv2, text, (xmin + 5, ymin), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    
    image_cv2 = cv2.cvtColor(image_cv2, cv2.COLOR_BGR2RGB)

    plt.imshow(image_cv2)
    plt.axis('off')
    if save_name:
        plt.savefig(save_name, bbox_inches='tight')
    plt.show()

def annotate_mask_object(row: pd.Series, label: str, save_name: Optional[str] = None):
    """
    Annotate an image with mask and bounding box information for a specified object label.

    Args:
    - row (pd.Series): Row from a DataFrame containing the following columns:
                       - 'image': Image data (as a PIL image or np.ndarray).
                       - 'box': Bounding box coordinates [xmin, ymin, xmax, ymax].
                       - 'mask': Segmentation mask as a 2D numpy array.
                       - 'obj_idx': Index of the object.
    - label (str): Label of the object to annotate (e.g., "fire hydrant").
    - save_name (Optional[str]): Optional path to save the annotated image. Defaults to None.

    Returns:
    - None: Displays the annotated image and optionally saves it if `save_name` is specified.
    """
    plt.figure(figsize=(10, 6))
    ax = plt.gca()
    image = row['image'].image
    image_cv2 = np.array(image) if isinstance(image, Image.Image) else image
    #image_cv2 = cv2.cvtColor(image_cv2, cv2.COLOR_RGB2BGR)

    padding = 15
    image_cv2 = cv2.copyMakeBorder(image_cv2, padding, padding, padding, padding, cv2.BORDER_CONSTANT, value=[255, 255, 255])

    color_name = random_named_css_colors(1)[0]

    xmin, ymin, _, _ = row['box']
    xmin += padding
    ymin += padding

    mask = np.array(row['mask'])
    mask_uint8 = (mask * 255).astype(np.uint8)
    contours, _ = cv2.findContours(mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    color = webcolors.name_to_rgb(color_name)
    color = (color.red, color.green, color.blue)
    cv2.drawContours(image_cv2, [c + [padding, padding] for c in contours], -1, color, 2)

    obj_idx = int(row["obj_idx"])
    text = f'{label} [{obj_idx}]'
    (text_width, text_height), baseline = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
    cv2.rectangle(image_cv2, (xmin + 5, ymin - text_height - baseline), 
                  (xmin + 5 + text_width, ymin + baseline), color, thickness=cv2.FILLED)
    cv2.putText(image_cv2, text, (xmin + 5, ymin), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    
    image_cv2 = cv2.cvtColor(image_cv2, cv2.COLOR_BGR2RGB)

    plt.imshow(image_cv2)
    plt.axis('off')
    if save_name:
        plt.savefig(save_name, bbox_inches='tight')
    plt.show()
    
def estimate_locations(dists: pd.DataFrame, index: int, label1: str, label2: str) -> None:
    """
    Visualize the closest points between two sets of coordinates on the X-Z plane, linking them with lines
    and annotating their distances for a specific image index.

    Args:
    - dists (pd.DataFrame): DataFrame containing distances and coordinate information.
    - index (int): The specific image index to filter the DataFrame and visualize the results.
    - label1 (str): The label for the first set of coordinates.
    - label2 (str): The label for the second set of coordinates.

    Returns:
    - None
    """
    df = dists[dists['img_idx'] == index]
    
    fig, ax = plt.subplots()
    all_coords = []
    annotations = []

    for _, row in df.iterrows():
        coords1 = np.array([[coord[0], coord[2]] for coord in row[f"coords_{label1}"]])
        coords2 = np.array([[coord[0], coord[2]] for coord in row[f"coords_{label2}"]])
        distance, obj_index1, obj_index2 = row['distance'], row[f"obj_idx_{label1}"], row[f"obj_idx_{label2}"]
        
        all_coords.extend(coords1)
        all_coords.extend(coords2)
        
        dist_matrix = np.linalg.norm(coords1[:, np.newaxis] - coords2, axis=2)
        min_dist_idx = np.unravel_index(np.argmin(dist_matrix), dist_matrix.shape)
        point1, point2 = coords1[min_dist_idx[0]], coords2[min_dist_idx[1]]

        ax.plot(*point1, 'go', label=label1 if _ == 0 else "")
        ax.plot(*point2, 'bo', label=label2 if _ == 0 else "")

        ax.plot(*zip(point1, point2), 'b--')

        ax.text(*point1, f'{obj_index1}', fontsize=12, color='green')
        ax.text(*point2, f'{obj_index2}', fontsize=12, color='red')

        mid_point = np.mean([point1, point2], axis=0)

        offset_y = 10
        for ann in annotations:
            if np.abs(ann[0] - mid_point[0]) < 0.1:  
                offset_y += 15  

        annotation = ax.annotate(f'{distance:.5f}', xy=mid_point, textcoords="offset points",
                                 xytext=(0, offset_y), ha='center', fontsize=10, color='blue',
                                 bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="blue", lw=0, alpha=0))
        annotations.append((mid_point[0], mid_point[1] + offset_y))

    all_coords = np.array(all_coords)
    padding = 1
    ax.set_xlim(all_coords[:, 0].min() - padding, all_coords[:, 0].max() + padding)
    ax.set_ylim(all_coords[:, 1].min() - padding, all_coords[:, 1].max() + padding)

    plt.legend()
    plt.xlabel('X')
    plt.ylabel('Z')
    plt.title('Closest Points on X-Z Plane')
    plt.axis('equal')
    plt.show()

def estimate_box(dists: pd.DataFrame, index: int, label1: str, label2: str) -> None:
    """
    Visualize bounding boxes and the closest points between two sets of coordinates on the X-Z plane.

    Args:
    - dists (pd.DataFrame): DataFrame containing coordinates and distances. Must contain columns:
                            'img_idx', 'distance', 'coords_{label1}', 'coords_{label2}',
                            'obj_idx_{label1}', and 'obj_idx_{label2}'.
    - index (int): Specific image index to filter the DataFrame and visualize results for.
    - label1 (str): Label for the first set of coordinates (e.g., "house").
    - label2 (str): Label for the second set of coordinates (e.g., "tree").

    Returns:
    - None: Displays the plot of bounding boxes and closest points.
    """
    df = dists[dists['img_idx'] == index]
    
    fig, ax = plt.subplots()
    all_coords = []

    label1_box_plotted = False
    label2_box_plotted = False

    for _, row in df.iterrows():
        coords1 = np.array([[coord[0], coord[2]] for coord in row[f"coords_{label1}"]])
        coords2 = np.array([[coord[0], coord[2]] for coord in row[f"coords_{label2}"]])
        distance, obj_index1, obj_index2 = row['distance'], row[f"obj_idx_{label1}"], row[f"obj_idx_{label2}"]
        
        # Bounding boxes for label1 and label2 on X-Z plane
        min_x1, min_z1 = coords1.min(axis=0)
        max_x1, max_z1 = coords1.max(axis=0)
        min_x2, min_z2 = coords2.min(axis=0)
        max_x2, max_z2 = coords2.max(axis=0)

        all_coords.extend([[min_x1, min_z1], [max_x1, max_z1], [min_x2, min_z2], [max_x2, max_z2]])

        # Plot bounding boxes
        if not label1_box_plotted:
            ax.plot([min_x1, max_x1, max_x1, min_x1, min_x1], [min_z1, min_z1, max_z1, max_z1, min_z1], 
                    'g-', label=f"{label1} Box")
            label1_box_plotted = True
        else:
            ax.plot([min_x1, max_x1, max_x1, min_x1, min_x1], [min_z1, min_z1, max_z1, max_z1, min_z1], 'g-')
        
        if not label2_box_plotted:
            ax.plot([min_x2, max_x2, max_x2, min_x2, min_x2], [min_z2, min_z2, max_z2, max_z2, min_z2], 
                    'b-', label=f"{label2} Box")
            label2_box_plotted = True
        else:
            ax.plot([min_x2, max_x2, max_x2, min_x2, min_x2], [min_z2, min_z2, max_z2, max_z2, min_z2], 'b-')

        # Find centers of the bounding boxes
        center1 = [(min_x1 + max_x1) / 2, (min_z1 + max_z1) / 2]
        center2 = [(min_x2 + max_x2) / 2, (min_z2 + max_z2) / 2]

        # Annotate object indices at the bounding box centers with a cleaner style
        ax.text(center1[0], center1[1], f'{obj_index1}', fontsize=8, color='green', ha='center', va='center')
        ax.text(center2[0], center2[1], f'{obj_index2}', fontsize=8, color='blue', ha='center', va='center')

        # Find and plot the closest points within the bounding boxes
        dist_matrix = np.linalg.norm(coords1[:, np.newaxis] - coords2, axis=2)
        min_dist_idx = np.unravel_index(np.argmin(dist_matrix), dist_matrix.shape)
        point1, point2 = coords1[min_dist_idx[0]], coords2[min_dist_idx[1]]

        # Link closest points with a dashed line
        ax.plot(*zip(point1, point2), 'k--')

        # Annotate distance at the midpoint between closest points
        mid_point = np.mean([point1, point2], axis=0)
        ax.annotate(f'{distance:.5f}', xy=mid_point, textcoords="offset points",
                    xytext=(-7.5, 7.5), ha='center', fontsize=10, color='red',
                    bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="blue", lw=0, alpha=0))

    all_coords = np.array(all_coords)

    if all_coords.size > 0:
        padding = 1
        ax.set_xlim(all_coords[:, 0].min() - padding, all_coords[:, 0].max() + padding)
        ax.set_ylim(all_coords[:, 1].min() - padding, all_coords[:, 1].max() + padding)

    plt.legend(loc='upper left', bbox_to_anchor=(1.05, 1), borderaxespad=0.)
    plt.xlabel('X')
    plt.ylabel('Z')
    plt.title('Bounding Boxes and Closest Points on X-Z Plane')
    plt.axis('equal')
    plt.show()


def plot_results(pil_img, scores: List[float], labels: List[str], boxes: List[List[int]]) -> None:
    """
    Plots detection results on an image.

    Args:
    - pil_img (Image.Image): The image to be plotted.
    - scores (List[float]): List of scores for each detection.
    - labels (List[str]): List of labels for each detection.
    - boxes (List[List[int]]): List of bounding boxes for each detection.

    Returns:
    - None
    """
    plt.figure(figsize=(16, 10))
    plt.imshow(pil_img)
    ax = plt.gca()
    colors = COLORS * 100
    for score, label, (xmin, ymin, xmax, ymax), c in zip(scores, labels, boxes, colors):
        ax.add_patch(plt.Rectangle((xmin, ymin), xmax - xmin, ymax - ymin,
                                   fill=False, color=c, linewidth=3))
        label_text = f'{label}: {score:0.2f}'
        ax.text(xmin, ymin, label_text, fontsize=15,
                bbox=dict(facecolor='none', alpha=0.5))
    plt.axis('off')
    plt.show()

def plot_detections(
    image: Union[Image.Image, np.ndarray],
    detections: List[DetectionResult],
    save_name: Optional[str] = None
) -> None:
    """
    Plots detections on an image.

    Args:
    - image (Union[Image.Image, np.ndarray]): The image to be plotted.
    - detections (List[DetectionResult]): List of detection results.
    - save_name (Optional[str]): Path to save the plotted image. Defaults to None.

    Returns:
    - None
    """
    annotated_image = annotate(image, detections)
    plt.imshow(annotated_image)
    plt.axis('off')
    if save_name:
        plt.savefig(save_name, bbox_inches='tight')
    plt.show()
     
def random_named_css_colors(num_colors: int) -> List[str]:
    """
    Generate a list of random CSS color names.

    Args:
    - num_colors (int): Number of random colors to generate.

    Returns:
    - List[str]: List of randomly selected CSS color names as strings.
    """
    colors = ['aqua', 'black', 'blue', 'blueviolet', 'brown', 'burlywood', 'cadetblue', 'chartreuse', 'chocolate', 'coral', 
             'cornflowerblue', 'crimson', 'cyan', 'darkblue', 'darkcyan', 'darkgoldenrod', 'darkgray', 'darkgreen',
             'darkgrey', 'darkkhaki', 'darkmagenta', 'darkolivegreen', 'darkorange', 'darkorchid', 'darkred', 'darksalmon',
             'darkseagreen', 'darkslateblue', 'darkslategray', 'darkslategrey', 'darkturquoise', 'darkviolet', 'deeppink', 
             'deepskyblue', 'dimgray', 'dimgrey', 'dodgerblue', 'firebrick', 'forestgreen', 'fuchsia', 'goldenrod', 'gray', 
             'green', 'grey', 'hotpink', 'indianred', 'indigo', 'lawngreen', 'lightcoral', 'lightsalmon', 
             'lightseagreen',  'lightslategray', 'lightslategrey', 'lime', 'limegreen', 
             'magenta', 'maroon', 'mediumaquamarine', 'mediumblue', 'mediumorchid', 'mediumpurple', 'mediumseagreen', 
             'mediumslateblue', 'mediumspringgreen', 'mediumturquoise', 'mediumvioletred', 'midnightblue', 'navy', 'olive', 
             'olivedrab', 'orange', 'orangered', 'orchid', 'palevioletred', 'peru', 'plum', 'purple', 'red', 'rosybrown', 
             'royalblue', 'saddlebrown', 'salmon', 'sandybrown', 'seagreen', 'sienna', 'silver', 'skyblue', 'slateblue', 
             'slategray', 'slategrey', 'springgreen', 'steelblue', 'tan', 'teal', 'tomato', 'turquoise', 'violet', 'yellowgreen']

    return random.sample(colors, min(num_colors, len(colors)))

def nearest_object_existence(
    gdf: gpd.GeoDataFrame, objs: gpd.GeoDataFrame, meta: pd.DataFrame, images: pd.DataFrame,
    max_dist: float = 0.001, visualize: bool = False
) -> pd.DataFrame:
    """
    Check if nearest objects exist within a specified distance and optionally visualize.

    Args:
    - gdf (gpd.GeoDataFrame): GeoDataFrame containing spatial geometries to check against.
                              Must contain a 'geometry' column with Point geometries.
    - objs (gpd.GeoDataFrame): GeoDataFrame containing object geometries with columns:
                               - 'img_idx': Index linking to images.
                               - 'meta_pt': Point geometries.
    - meta (pd.DataFrame): DataFrame with metadata for each image, including:
                           - 'meta_pt': Point geometries indicating locations.
    - images (pd.DataFrame): DataFrame with image data including:
                             - 'meta_pt': Point geometries.
                             - 'perp_heading': Heading of the image.
                             - 'image': Image data or file path.
    - max_dist (float): Maximum distance to consider for finding nearest objects.
    - visualize (bool): Whether to visualize the results with bounding circles.

    Returns:
    - pd.DataFrame: DataFrame indicating the existence of nearest objects, with the following columns:
                    - 'geometry': Original geometry from gdf.
                    - 'img_idx': Index of the nearest image.
                    - 'obj_idx': Index of the nearest object.
                    - 'label', 'conf', 'box', 'mask', 'coords', 'distance': Nearest object details.
                    - 'meta_pt', 'perp_heading', 'image': Metadata and image info from the nearest match.
    """
    # Merge objects with metadata and set the geometry
    obj_meta = pd.merge(objs, meta['meta_pt'][~meta['meta_pt'].index.duplicated(keep='first')], left_on='img_idx', right_index=True)
    obj_meta = gpd.GeoDataFrame(obj_meta).set_geometry('meta_pt')
    
    # Perform nearest spatial join with distance column
    joined = gpd.sjoin_nearest(gdf, obj_meta, how='left', max_distance=max_dist, distance_col="distance")
    nearest_exist = ~joined['index_right'].isna()
    
    # Filter for rows with valid nearest objects
    nearest_df = joined[nearest_exist]
    
    # Join with images on `img_idx` from nearest_df and `index` from images
    result = pd.merge(nearest_df, images, how='left', left_on='img_idx', right_index=True, suffixes=('', '_image'))
    
    if visualize: 
        fig, ax = plt.subplots(figsize=(8, 8))
        
        # Plot the points from gdf and obj_meta
        gdf.plot(ax=ax, color='blue', markersize=10, label='GDF Points')
        obj_meta.plot(ax=ax, color='orange', markersize=10, label='Objects')
        
        # Only draw circles for points that have valid nearest objects within max_dist
        for idx, row in nearest_df.iterrows():
            if row['distance'] <= max_dist:
                point = row['geometry']
                circle = Circle((point.x, point.y), max_dist, color='red', fill=False, linewidth=1)
                ax.add_patch(circle)
        
        # Optionally, plot meta as well if needed
        gpd.GeoDataFrame(meta).plot(ax=ax, linewidth=0.5, label='Meta Data')
        
        # Set legend and title
        plt.legend()
        plt.title('Nearest Objects within Specified Distance')
        plt.xlabel('Longitude')
        plt.ylabel('Latitude')
        
    return result[['geometry', 'img_idx', 'obj_idx', 'label', 'conf', 'box',
       'mask', 'coords', 'distance', 'meta_pt', 'perp_heading', 'image']]

def nearest_image_existence(
    gdf: gpd.GeoDataFrame, images: gpd.GeoDataFrame,
    max_dist: float = 0.0001, visualize: bool = False
) -> pd.DataFrame:
    """
    Check if nearest images exist within a specified distance and optionally visualize.

    Args:
    - gdf (gpd.GeoDataFrame): GeoDataFrame containing geometries.
    - images (gpd.GeoDataFrame): GeoDataFrame containing image data.
    - meta (pd.DataFrame): DataFrame containing metadata.
    - max_dist (float): Maximum distance to consider for nearest images.
    - visualize (bool): Whether to visualize the results.

    Returns:
    - pd.DataFrame: DataFrame indicating the existence of nearest images, joined with metadata.
    """
    
    images = images[['meta_pt', 'perp_heading', 'image']].set_geometry('meta_pt')
    
    # Perform nearest spatial join with distance column
    joined = gpd.sjoin_nearest(gdf, images, how='left', max_distance=max_dist, distance_col="distance")
    nearest_exist = ~joined['index_right'].isna()
    
    # Filter for rows with valid nearest images
    nearest_df = joined[nearest_exist]
    nearest_df.rename(columns={'index_right': 'img_idx'}, inplace=True)
        
    return nearest_df
    
def estimate_geolocations(bounds: List[Tuple[float, float, float]], img_loc: Point, heading: float, lat_scale: float = 111320) -> Point:
    """
    Estimate the geographic location of an object based on its bounding box coordinates, image location, and heading.

    Args:
    - bounds (List[Tuple[float, float, float]]): List of tuples representing each corner of a 3D bounding box in local coordinates (x, y, z).
    - img_loc (Point): Geographic location of the image in longitude and latitude.
    - heading (float): Heading angle in degrees from the image location.
    - lat_scale (float): Scaling factor for converting latitude degrees to meters. Default is 111320.

    Returns:
    - Point: Estimated geographic location of the object as a shapely Point (longitude, latitude).
    """
    lon, lat = img_loc.x, img_loc.y
    lon_scale = lat_scale * np.cos(np.radians(lat))

    x, y, z = np.mean(bounds, axis=0)
    heading_rad = math.radians(heading)
    
    delta_lat = z / lat_scale * np.cos(heading_rad) - x / lon_scale * np.sin(heading_rad)
    delta_lon = x / lon_scale * np.cos(heading_rad) + z / lat_scale * np.sin(heading_rad)
    
    return Point(lon + delta_lon, lat + delta_lat)

def estimate_object_locations(bounds: pd.DataFrame, meta: pd.DataFrame, visualize: bool = False) -> gpd.GeoSeries:
    """
    Estimate geographic locations for each object using bounding box data and metadata.

    Args:
    - bounds (pd.DataFrame): DataFrame containing bounding box data with columns:
                             - 'img_idx': Index of the image.
                             - 'coords': List of 3D coordinates for bounding box corners.
                             - 'label': Label of the detected object.
    - meta (pd.DataFrame): DataFrame containing metadata for each image, including:
                           - 'meta_pt': Point geometry for each image location.
                           - 'heading': Heading in degrees.
    - visualize (bool): Whether to visualize the estimated geographic locations on a plot.

    Returns:
    - gpd.GeoSeries: GeoSeries where each Point represents the estimated geographic location of an object.
    """
    bounds = bounds.merge(meta[['meta_pt', 'heading']], left_on='img_idx', right_index=True)
    geolocs = bounds.apply(lambda row: estimate_geolocations(row['coords'], row['meta_pt'], row['heading']), axis=1)

    if visualize:
        fig, ax = plt.subplots(figsize=(8, 8))
        unique_labels = bounds['label'].unique()
        cmap = get_cmap('tab10') 

        for idx, label in enumerate(unique_labels):
            label_geolocs = gpd.GeoSeries([geoloc for geoloc, lbl in zip(geolocs, bounds['label']) if lbl == label])
            label_geolocs.plot(ax=ax, markersize=5, color=cmap(idx), label=label)

        gpd.GeoSeries(meta['geometry']).plot(ax=ax, markersize=5)
        plt.title("Estimate Geographic Locations of Detected Objects")
        plt.legend(loc='upper left', bbox_to_anchor=(1.05, 1), borderaxespad=0.)
        plt.show()

    return gpd.GeoSeries(geolocs)
    
def new_position(point: Point, heading: float, distance: float) -> Point:
    """
    Calculate a new position from a point given a heading and distance.

    Args:
    - point (Point): Original point.
    - heading (float): Heading angle.
    - distance (float): Distance to move.

    Returns:
    - Point: New position.
    """
    heading_rad = math.radians(heading)
    delta_y = distance * math.sin(heading_rad)
    delta_x = distance * math.cos(heading_rad)
    return Point(point.x + delta_x, point.y + delta_y)
