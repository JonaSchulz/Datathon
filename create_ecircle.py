import cv2
import numpy as np
import os
import matplotlib.pyplot as plt
import matplotlib.patheffects as PathEffects  
import networkx as nx



def preprocess_mask(mask):
    kernel = np.ones((3, 3), np.uint8)
    processed_mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    return processed_mask

def fit_ellipse_to_circle(mask):
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        largest_contour = max(contours, key=cv2.contourArea)
        ellipse = cv2.fitEllipse(largest_contour)
        return ellipse
    return None

def check_contour_alignment(mask, ellipse, contour):
    # Calculate the parameters of the ellipse
    (xc, yc), (d1, d2), angle = ellipse
    center = np.array([xc, yc])
    radius = max(d1, d2) / 2

    aligned_count = 0
    total_points = 0

    # Calculate distance of each contour point to the ellipse's center and check if it lies close to the boundary
    for point in contour.squeeze():
        point_distance = np.linalg.norm(point - center)
        if abs(point_distance - radius) < 10:  # Threshold of 10 pixels
            aligned_count += 1
        total_points += 1

    # Check if a significant portion of the points are aligned
    if aligned_count / total_points > 0.01:  # 20% of the points need to be aligned
        ellipse_img = np.zeros_like(mask)
        cv2.ellipse(ellipse_img, ellipse, (255, 255, 255), 2)  # White ellipse
        # Draw the mask contour
        #cv2.drawContours(ellipse_img, [contour], -1, (0, 255, 0), 2)  # Green contour
        #plt.figure(figsize=(10, 10))
        #plt.imshow(ellipse_img, cmap='gray')
        #plt.title('Ellipse and Contour Overlay')
        #plt.show()
        return True
    return False


def visualize_contour_alignment(mask, ellipse, contour):
    # Create an image to draw the ellipse
    ellipse_img = np.zeros_like(mask)
    cv2.ellipse(ellipse_img, ellipse, (255, 255, 255), 2)  # White ellipse

    # Draw the mask contour
    cv2.drawContours(ellipse_img, [contour], -1, (0, 255, 0), 2)  # Green contour

    # Use matplotlib to display
    plt.figure(figsize=(10, 10))
    plt.imshow(ellipse_img, cmap='gray')
    plt.title('Ellipse and Contour Overlay')
    plt.show()

def analyze_shape(mask):
    processed_mask = preprocess_mask(mask)
    contours, _ = cv2.findContours(processed_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if contours:
        largest_contour = max(contours, key=cv2.contourArea)
        hull = cv2.convexHull(largest_contour, returnPoints=False)
        defects = cv2.convexityDefects(largest_contour, hull)

        if defects is not None and len(defects) > 0:  # Check if there are major defects
            # Analyze defects to determine if likely two slices or irregular object
            for i in range(defects.shape[0]):
                s, e, f, d = defects[i, 0]
                if d > 1000:  # d is the approximate distance between the farthest point and the convex hull
                    return False  # Found a significant defect, likely not a single slice

        # Optionally check aspect ratio and area ratio for further validation
        rect = cv2.minAreaRect(largest_contour)
        width, height = rect[1]
        aspect_ratio = max(width, height) / min(width, height)
        area_contour = cv2.contourArea(largest_contour)
        area_rect = width * height
        area_ratio = area_contour / area_rect

        if aspect_ratio > 2 or area_ratio < 0.5:
            return False  # Shape is too elongated or too much empty space in bounding rect

        return True  # Passed all checks, likely a single pie slice

    return False  # No contours found


def calculate_area_percentage(inner_mask, outer_mask):
    inner_area = cv2.countNonZero(inner_mask)
    outer_area = cv2.countNonZero(outer_mask)
    if outer_area == 0:
        return 0
    return (inner_area / outer_area) * 100

def percentage_overlap(mask, circle_mask):
    """Calculate the percentage of mask that overlaps with the circle mask."""
    overlap_area = cv2.bitwise_and(mask, mask, mask=circle_mask)
    total_area = np.count_nonzero(mask)
    overlap_area_count = np.count_nonzero(overlap_area)

    if total_area == 0:
        return 0  # Avoid division by zero if mask is empty

    return (overlap_area_count / total_area) * 100


def contour_overlaps(contour_a, contour_b):
    """ Determine if contour A overlaps significantly with contour B. """
    # Create masks from contours
    mask_a = np.zeros((1000, 1000), dtype=np.uint8)  # Example size, adjust to your needs
    mask_b = np.zeros((1000, 1000), dtype=np.uint8)
    cv2.drawContours(mask_a, [contour_a], -1, (255), thickness=cv2.FILLED)
    cv2.drawContours(mask_b, [contour_b], -1, (255), thickness=cv2.FILLED)

    # Calculate overlap
    intersection = np.logical_and(mask_a, mask_b)
    if np.sum(intersection) / np.sum(mask_b) > 0.8:  # 80% of B is covered by A
        return True
    return False

def calculate_overlap(mask1, mask2):
    """Calculate the area of overlap between two masks."""
    intersection = cv2.bitwise_and(mask1, mask2)
    overlap_area = np.count_nonzero(intersection)
    return overlap_area

def remove_overlapping_masks(masks, paths):
    """Remove larger masks that cover smaller masks significantly."""
    # Sort masks by area (smallest first to protect smaller slices)
    masks_sorted = sorted(masks, key=lambda x: np.count_nonzero(x[1]))
    paths_sorted = sorted(paths, key=lambda x: np.count_nonzero(x[1]))
    to_remove = set()

    # Compare each mask with every other mask
    for i in range(len(masks_sorted)):
        for j in range(len(masks_sorted)):
            if i != j and i not in to_remove and j not in to_remove:
                mask1_area = np.count_nonzero(masks_sorted[i])
                mask2_area = np.count_nonzero(masks_sorted[j])
                overlap_area = calculate_overlap(masks_sorted[i], masks_sorted[j])

                # Check if one mask covers more than 90% of the other
                if overlap_area / (mask1_area + 0.0001) > 0.9:
                    # Remove the larger mask if it covers a smaller mask significantly
                    if mask1_area < mask2_area:
                        to_remove.add(j)
                    else:
                        to_remove.add(i)

    # Filter out the masks marked for removal
    filtered_masks = [masks_sorted[i] for i in range(len(masks_sorted)) if i not in to_remove]
    filtered_paths = [paths_sorted[i] for i in range(len(paths_sorted)) if i not in to_remove]
    return filtered_masks, filtered_paths


def process_folder(folder_path):
    masks = [os.path.join(folder_path, f) for f in os.listdir(folder_path) if f.endswith('.png')]
    results = []

    for mask_path in masks:
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        if mask is None:
            continue
        is_circle, ellipse = is_circle_and_draw(image_path,mask_path)
        if is_circle:
            circle_mask = mask  # The mask of the circle
            circle_path = mask_path
            break
    else:
        print("No circular masks found in the folder.")
        return results
    print("masks before filtering them: "+str(len(masks)))
    mask_list = []
    mask_paths = []
    for mask_path in masks:
        if mask_path == circle_path:  # Skip the circle mask itself
            continue

        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        
        
        """
        if mask is None or not analyze_shape(mask):
            print("Non convex surface found")
            cv2.imshow('Non convex surface', cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR))
            cv2.waitKey(0)
            cv2.destroyAllWindows()
            continue
        
                """
        #contour_list = []
        if ellipse and mask is not None:
            slice_mask = preprocess_mask(mask)
            contours, _ = cv2.findContours(slice_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if contours:
                largest_contour = max(contours, key=cv2.contourArea)
                if not check_contour_alignment(slice_mask, ellipse, largest_contour):
                    continue
                else:
                    #contour_list.append(largest_contour)
                    mask_list.append(mask)
                    mask_paths.append(mask_path)
            else:
                print("No contours found in slice mask.")
            
            """if is_slice(slice_mask, ellipse):
                print("The mask is likely a pie slice.")
                cv2.imshow('The mask is likely a pie slice.', cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR))
                cv2.waitKey(0)
                cv2.destroyAllWindows()
            else:
                cv2.imshow("The mask is likely not a pie slice.", cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR))
                cv2.waitKey(0)
                cv2.destroyAllWindows()
                print("The mask is likely not a pie slice.") 
                continue"""
    print("masks after circle filtering them: "+str(len(mask_list))) 
    filtered_masks, filtered_paths = remove_overlapping_masks(mask_list, mask_paths)  
    #filtered_masks = mask_list
    print("masks after overlap filtering them: "+str(len(filtered_masks))) 
    for i,mask in enumerate(filtered_masks):
        overlap_percentage = percentage_overlap(mask, circle_mask)
        if overlap_percentage >= 0.95:
            # Calculate the area percentage
            percentage = calculate_area_percentage(mask, circle_mask)
            results.append((filtered_paths[i], percentage))

    return results, circle_path

def is_circle(mask, circularity_threshold=0.95):
    # Find contours in the mask
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        # Find the largest contour based on area
        largest_contour = max(contours, key=cv2.contourArea)
        if cv2.contourArea(largest_contour) > 100:  # Check if the contour is large enough
            # Fit an ellipse to the contour
            ellipse = cv2.fitEllipse(largest_contour)

            # Extract the axes of the ellipse
            (center, axes, orientation) = ellipse
            major_axis = max(axes)
            minor_axis = min(axes)

            # Calculate the ratio of the axes
            axis_ratio = minor_axis / major_axis
            # Check if the axes are nearly equal
            return axis_ratio > circularity_threshold

    return False

def is_circle_and_draw(image_path, mask_path, circularity_threshold=0.95):
    # Load the original image and mask
    image = cv2.imread(image_path)
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

    if image is None or mask is None:
        print("Image or mask not found or invalid format.")
        return False, None

    # Find contours in the mask
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        # Find the largest contour based on area
        largest_contour = max(contours, key=cv2.contourArea)
        if cv2.contourArea(largest_contour) > 100:  # Check if the contour is large enough
            # Fit an ellipse to the contour
            ellipse = cv2.fitEllipse(largest_contour)

            # Draw the ellipse on the mask
            ellipse_image = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)  # Convert mask to BGR to draw colored ellipse
            cv2.ellipse(ellipse_image, ellipse, (0, 255, 0), 2)  # Green ellipse with a thickness of 2

            # Calculate the axis ratio to check if it's a circle
            (center, axes, orientation) = ellipse
            major_axis = max(axes)
            minor_axis = min(axes)
            axis_ratio = minor_axis / major_axis

            # Determine if it's a circle based on the axis ratio
            is_circle = axis_ratio > circularity_threshold

            # Show the result
            if is_circle:
                cv2.imshow('Mask with Ellipse', ellipse_image)
                cv2.waitKey(0)
                cv2.destroyAllWindows()

            return is_circle, ellipse

    return False, None
def visualize_segment_masks(folder_path, results):
    # Define colors for different masks
    colors = [
        (255, 0, 0), (0, 255, 0), (0, 0, 255),
        (255, 255, 0), (0, 255, 255), (255, 0, 255),
        (192, 192, 192), (128, 128, 0), (128, 0, 128),
        (0, 128, 128), (64, 64, 64), (64, 0, 0)
    ]  # Add more colors if needed

    fig, ax = plt.subplots(figsize=(10, 10))
    # Create a blank canvas, assume the first mask to determine size
    if results:
        first_mask = cv2.imread(results[0][0], cv2.IMREAD_GRAYSCALE)
        if first_mask is None:
            print("Failed to load the first mask image.")
            return
        canvas = np.zeros((first_mask.shape[0], first_mask.shape[1], 3), dtype=np.uint8)
    else:
        print("No results to display.")
        return

    for idx, (mask_path, percentage) in enumerate(results):
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        if mask is None:
            continue

        color = colors[idx % len(colors)]
        # Apply color to the mask
        for i in range(3):  # Apply the color to the 3 channels
            canvas[:, :, i] = np.where(mask == 255, color[i], canvas[:, :, i])

        # Calculate position for text (centroid of the mask)
        M = cv2.moments(mask)
        if M['m00'] != 0:
            cx = int(M['m10'] / M['m00'])
            cy = int(M['m01'] / M['m00'])
            # Plotting text with a dark contour for readability
            ax.text(cx, cy, f"{percentage:.2f}%", color='white', fontsize=12, ha='center', path_effects=[PathEffects.withStroke(linewidth=3, foreground='black')])

    ax.imshow(canvas)
    ax.axis('off')  # Hide the axes
    plt.show()

image_path = 'input\\chart_77.png'
mask_path = 'output\\chart_77\\5.png'

"""
mask = cv2.imread('output\\unnamed\\2.png', cv2.IMREAD_GRAYSCALE)  # Ensure this is a grayscale image
if mask is not None:
    if is_circle(mask):
        print("The mask is circular.")
        _, ellipse_image = is_circle_and_draw(image_path, mask_path)

    
    
    else:
        print("The mask is not circular.")
        _, ellipse_image = is_circle_and_draw(image_path, mask_path)
else:
    print("Image not found or invalid image format.")
"""
folder_path = 'output\\chart_67'
results, circle_path = process_folder(folder_path)
for result in results:
    print(f"Mask {result[0]} covers {result[1]:.2f}% of the circle area.")

visualize_segment_masks(folder_path, results)


