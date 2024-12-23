import os
import numpy as np
import copy
import cv2
from collections import Counter
from skimage.measure import label, regionprops
import glob
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

import os
os.environ["QT_QPA_PLATFORM"] = "offscreen"
import matplotlib
matplotlib.use('TkAgg') 

def overlay_segmentation_result(cropped_image: np.ndarray, mask: np.ndarray, colors: dict) -> np.ndarray:
    """
    분할 결과를 크롭된 이미지 위에 오버레이합니다.

    Args:
    cropped_image: 크롭된 이미지
    mask: 분할 마스크
    colors: 클래스 레이블을 색상에 매핑하는 딕셔너리

    Returns:
    오버레이된 이미지
    """
    try:
        img_array = np.array(cropped_image)
        _img_array = copy.deepcopy(img_array)
        mask_rgb = np.zeros((_img_array.shape[0], _img_array.shape[1], 3), dtype=np.uint8)

        for class_label, color in colors.items():
            mask_rgb[mask == class_label] = color

        alpha = 0.5  # Transparency factor
        overlay = cv2.addWeighted(_img_array, alpha, mask_rgb, 1 - alpha, 0)
        return overlay
    except Exception as e:
        raise RuntimeError(f"Error overlaying segmentation result: {e}")

def __calculate_thickness(region_mask: np.ndarray, pixel_resolution: float = 0.5) -> tuple[list[float], list[tuple[int, int, int]]]:
    """
    유간 계산 함수.
    (2024.09.10) 유간 계산 시 y축 좌표가 이미지의 가장자리에 해당하는 경우 실제 유간이 아닌 것으로 판단하여 수정함 해당 함수는 구버전 

    Args:
    region_mask: 영역 마스크
    pixel_resolution: 픽셀 해상도 (기본값: 0.5mm)

    Returns:
    label, regionprops에 따라 분리된 영역 마스크의 두께에서 계산한 각 열 좌표 별 두께 및 두께를 계산하는데 사용된 행의 좌표
    """
    if region_mask.ndim != 2:
        raise ValueError("region_mask must be a 2D array")
    
    rows, cols = region_mask.shape
    thicknesses = []
    thickness_positions = []
    
    for x in range(cols):
        col_mask = region_mask[:, x]
        if np.any(col_mask):
            y_coords = np.where(col_mask)[0]
            y_min, y_max = np.min(y_coords), np.max(y_coords)
            thickness = (y_max - y_min + 1) * pixel_resolution
            thicknesses.append(thickness)
            thickness_positions.append((x, y_min, y_max))
    
    return thicknesses, thickness_positions

def calculate_thickness(region_mask: np.ndarray, pixel_resolution: float = 0.5, edge_threshold: int = 5) -> tuple[list[float], list[tuple[int, int, int]]]:
    """
    영역의 두께를 계산하며, 경사와 가장자리 문제를 고려합니다.

    Args:
    region_mask: 영역 마스크
    pixel_resolution: 픽셀 해상도 (기본값: 0.5mm)
    edge_threshold: 가장자리로 간주할 픽셀 수 (기본값: 3)

    Returns:
    각 열 좌표별 두께 및 두께를 계산하는 데 사용된 행의 좌표
    """
    if region_mask.ndim != 2:
        raise ValueError("region_mask must be a 2D array")
    
    rows, cols = region_mask.shape
    thicknesses = []
    thickness_positions = []
    
    for x in range(cols):
        col_mask = region_mask[:, x]
        if np.any(col_mask):
            y_coords = np.where(col_mask)[0]
            y_min, y_max = np.min(y_coords), np.max(y_coords)
            
            # 가장자리 체크
            if y_min < edge_threshold or y_max > rows - edge_threshold:
                continue
            
            # 경사 체크
            if y_max - y_min > 1:
                # 중간 지점 찾기
                mid_point = (y_min + y_max) // 2
                # 위쪽 경계 찾기
                upper_bound = mid_point
                while upper_bound > y_min and col_mask[upper_bound-1]:
                    upper_bound -= 1
                # 아래쪽 경계 찾기
                lower_bound = mid_point
                while lower_bound < y_max and col_mask[lower_bound+1]:
                    lower_bound += 1
                
                thickness = (lower_bound - upper_bound + 1) * pixel_resolution
                thicknesses.append(thickness)
                thickness_positions.append((x, upper_bound, lower_bound))
            else:
                thickness = (y_max - y_min + 1) * pixel_resolution
                thicknesses.append(thickness)
                thickness_positions.append((x, y_min, y_max))
    
    return thicknesses, thickness_positions


def calculate_mode(values: list[float], precision: int = 1) -> float:
    """
    주어진 값들의 최빈값을 계산합니다.

    Args:
    values: 값들의 리스트
    precision: 결과의 소수점 자리수 (기본값: 1)

    """
    rounded_values = np.round(values, precision)
    value_counts = Counter(rounded_values)
    mode = max(value_counts, key=value_counts.get)
    return mode
    

def gap_measure(mask: np.ndarray, image: np.ndarray, pixel_resolution: float=0.5, min_region_size: int=100):
    """
    주어진 마스크와 이미지를 사용하여 유간을 계산합니다.

    Args:
    mask: 원본 탐지 마스크  
    image: 원본 이미지
    pixel_resolution: 픽셀 해상도 (기본값: 0.5mm)
    min_region_size: 최소 영역 크기 (기본값: 100)

    Returns:
    두께 계산 결과
    """
    try:
        labeled_mask = label(mask, connectivity=1)
        regions = regionprops(labeled_mask)
        
        if not regions:
            return float('-inf')
        
        region_data = []
        all_thicknesses = []
        all_thickness_positions = []
        
        for region in regions:
            if region.area >= min_region_size:
                region_mask = labeled_mask == region.label
                thicknesses, thickness_positions = calculate_thickness(region_mask, pixel_resolution)
                
                if thicknesses:
                    mean_thickness = np.mean(thicknesses)
                    mode_thickness = calculate_mode(thicknesses)
                    
                    mean_positions = [pos for thickness, pos in zip(thicknesses, thickness_positions) if np.isclose(thickness, mean_thickness, atol = 0.1)]
                    mode_positions = [pos for thickness, pos in zip(thicknesses, thickness_positions) if np.isclose(thickness, mode_thickness, atol = 0.1)]
                    
                    thickness_cv = np.std(thicknesses) / mean_thickness if mean_thickness != 0 else float('inf')
                    
                    region_data.append({
                        'label': region.label,
                        'thicknesses': thicknesses,
                        'positions' : thickness_positions,
                        'mean_thickness' : mean_thickness,
                        'mode_thickness' : mode_thickness,
                        'mean_positions' : mean_positions,
                        'mode_positions' : mode_positions,
                        'cv' : thickness_cv
                    })
                    
                    all_thicknesses.extend(thicknesses)
                    all_thickness_positions.extend(thickness_positions)
        
        return mask, image, region_data, all_thicknesses, all_thickness_positions
    except Exception as e:
        raise RuntimeError(f"Error calculating thickness: {e}")

def visualize_total_thickness(region_data):
    all_region_thicknesses = [region['thicknesses'] for region in region_data]
    max_length = max(len(thicknesses) for thicknesses in all_region_thicknesses)
    
    padded_thicknesses = [np.pad(thicknesses, (0, max_length - len(thicknesses)), 'constant', constant_values=np.nan) 
                          for thicknesses in all_region_thicknesses]
    
    total_thickness = np.nansum(padded_thicknesses, axis=0)
    average_thickness = np.nanmean(padded_thicknesses, axis=0)

    df = pd.DataFrame({
        'X-axis': range(len(total_thickness)),
        'Total Thickness': total_thickness,
        'Average Thickness': average_thickness
    })

    mean_total = np.nanmean(total_thickness)
    mode_total = pd.Series(total_thickness).mode().values[0]
    mean_avg = np.nanmean(average_thickness)

    sns.set_style("darkgrid")
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(16, 16))

    palette = sns.color_palette("coolwarm", as_cmap=True)

    # Total Thickness plot
    sns.lineplot(x='X-axis', y='Total Thickness', data=df, linewidth=3, color=palette(0.8), ax=ax1)
    ax1.axhline(mean_total, color=palette(0.2), linestyle='--', linewidth=3, label=f'Mean: {mean_total:.2f}mm')
    ax1.axhline(mode_total, color=palette(0.8), linestyle=':', linewidth=3, label=f'Mode: {mode_total:.2f}mm')
    ax1.set_ylabel('Total Joint gap (mm)', fontsize=20)
    ax1.set_title('Total Joint Gap', fontsize=24)
    ax1.legend(fontsize=16, loc='best')
    ax1.tick_params(axis='both', which='major', labelsize=16)

    # Average Thickness plot
    sns.lineplot(x='X-axis', y='Average Thickness', data=df, linewidth=3, color=palette(0.4), ax=ax2)
    ax2.axhline(mean_avg, color=palette(0.6), linestyle='--', linewidth=3, label=f'Mean: {mean_avg:.2f}mm')
    ax2.set_ylabel('Average Joint gap (mm)', fontsize=20)
    ax2.set_xlabel('X-axis', fontsize=20)
    ax2.set_title('Average Joint Gap', fontsize=24)
    ax2.legend(fontsize=16, loc='best')
    ax2.tick_params(axis='both', which='major', labelsize=16)

    for ax in [ax1, ax2]:
        ax.set_facecolor('#f0f0f0')
        ax.grid(color='white', linestyle='-', linewidth=1)
        for spine in ax.spines.values():
            spine.set_visible(False)

    plt.tight_layout()
    
    # Add text annotations directly on the plot
    fig.text(0.01, 0.98, f"Total Mean: {mean_total:.2f}mm", fontsize=12, va='top')
    fig.text(0.01, 0.96, f"Total Mode: {mode_total:.2f}mm", fontsize=12, va='top')
    fig.text(0.01, 0.94, f"Average Mean: {mean_avg:.2f}mm", fontsize=12, va='top')
    
    # Show the plot
    plt.show()

def visualize_mask_thickness(original_image, mask, region_data):
    plt.figure(figsize=(16, 8))
    plt.imshow(cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB))
    plt.imshow(mask, alpha=0.3, cmap='jet')
    
    labeled_mask = label(mask)
    
    for region in region_data:
        color = np.random.rand(3,)
        
        for x, y_min, y_max in region['positions']:
            plt.plot([x, x], [y_min, y_max], color=color, alpha=0.3, linewidth=1)
        
        for x, y_min, y_max in region['mean_positions']:
            plt.plot([x, x], [y_min, y_max], color='yellow', linewidth=2)
        
        for x, y_min, y_max in region['mode_positions']:
            plt.plot([x, x], [y_min, y_max], color='red', linewidth=2)
        
        region_mask = labeled_mask == region['label']
        if np.any(region_mask):
            props = regionprops(region_mask.astype(int))
            if props:
                if props[0].area > 1500:
                    bbox = props[0].bbox
                    center_x = (bbox[1] + bbox[3]) / 2
                    if bbox[0] < mask.shape[0] / 2:
                        text_y = bbox[2] + 40
                        va = 'top'
                    else:
                        text_y = bbox[0] - 40
                        va = 'bottom'
                
                    plt.text(center_x, text_y, 
                             f"Region {region['label']}\nMean: {region['mean_thickness']:.2f}\nMode: {region['mode_thickness']:.2f}", 
                             color='white', fontsize=12, ha='center', va=va, 
                             bbox=dict(facecolor='black', alpha=0.5, edgecolor='none', pad=1))
    
    plt.axis('off')
    plt.tight_layout()
    plt.show()



if __name__ == "__main__":
    IMAGE_PATHS = sorted(glob.glob('/path/to/your/expansion joint image'))
    MASK_PATHS = sorted(glob.glob('/path/to/your/expansion joint mask'))

    if not IMAGE_PATHS or not MASK_PATHS:
        raise ValueError("No image or mask files found in the specified directories.")
    
    
    ind = np.random.randint(0, len(IMAGE_PATHS)-1)
    img_path, mask_path = IMAGE_PATHS[ind], MASK_PATHS[ind]
    if not os.path.exists(img_path) or not os.path.exists(mask_path):
        raise FileNotFoundError(f"Image or mask file does not exist: {img_path} or {mask_path}")
    image = cv2.imread(img_path)
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    gap_mask = (mask==2).astype(np.uint8)
    
    ## 유간 탐지 결과 확인을 원하면 아래 코드 주석 해제 후 시각화
    # gt_overlay = overlay_segmentation_result(image, gap_mask, colors={1: (0, 0, 255)})
    # # Resize the image before displaying
    # resized_gt_overlay = cv2.resize(gt_overlay, (800, 600))  # Adjust size as needed
    # cv2.imshow('gt_overlay', resized_gt_overlay)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    
    processed_mask, processed_image, region_data, all_thicknesses, all_thickness_positions = gap_measure(gap_mask, image)
    visualize_mask_thickness(processed_image, processed_mask, region_data)
    visualize_total_thickness(region_data)
 