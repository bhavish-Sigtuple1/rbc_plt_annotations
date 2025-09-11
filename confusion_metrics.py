import json
import os
import random
import time
import cv2
import numpy as np
import pandas as pd
import csv
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix

def calc_iou(box1, box2):
    x1, y1, w1, h1 = box1
    x2, y2, w2, h2 = box2

    x1_max = x1 + w1
    y1_max = y1 + h1
    x2_max = x2 + w2
    y2_max = y2 + h2

    inter_x1 = max(x1, x2)
    inter_y1 = max(y1, y2)
    inter_x2 = min(x1_max, x2_max)
    inter_y2 = min(y1_max, y2_max)
    inter_w = max(0, inter_x2 - inter_x1)
    inter_h = max(0, inter_y2 - inter_y1)
    inter_area = inter_w * inter_h

    box1_area = w1 * h1
    box2_area = w2 * h2
    union_area = box1_area + box2_area - inter_area

    return inter_area / union_area if union_area > 0 else 0

def evaluate_predictions(gt_path, pred_path, iou_threshold=0.5):
    """Evaluate predictions and compute Precision, Recall, F1-score."""
    with open(gt_path, 'r') as f:
        gt_data = json.load(f)
    with open(pred_path, 'r') as f:
        pred_data = json.load(f)
    
    gt_boxes = {ann['image_id']: [] for ann in gt_data['annotations']}
    for ann in gt_data['annotations']:
        gt_boxes[ann['image_id']].append((ann['bbox'], ann['category_id']))
    
    pred_boxes = {ann['image_id']: [] for ann in pred_data['annotations']}
    for ann in pred_data['annotations']:
        pred_boxes[ann['image_id']].append((ann['bbox'], ann['category_id']))
    
    tp, fp, fn = 0, 0, 0
    
    for image_id in gt_boxes:
        gt_bboxes = gt_boxes.get(image_id, [])
        pred_bboxes = pred_boxes.get(image_id, [])
        
        matched = set()
        
        for pred_bbox, pred_cat in pred_bboxes:
            best_iou = 0
            best_gt_idx = None
            
            for gt_idx, (gt_bbox, gt_cat) in enumerate(gt_bboxes):
                if gt_idx in matched or gt_cat != pred_cat:
                    continue
                
                iou = calc_iou(pred_bbox, gt_bbox)
                if iou > best_iou:
                    best_iou = iou
                    best_gt_idx = gt_idx
            
            if best_iou > iou_threshold and best_gt_idx is not None:
                tp += 1
                matched.add(best_gt_idx)
            else:
                fp += 1
        
        fn += len(gt_bboxes) - len(matched)
    
    precision = tp / (tp + fp) if tp + fp > 0 else 0
    recall = tp / (tp + fn) if tp + fn > 0 else 0
    f1_score = 2 * (precision * recall) / (precision + recall) if precision + recall > 0 else 0
    
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1 Score: {f1_score:.4f}")
    
    return precision, recall, f1_score


def evaluate_predictions_per_class(gt_path, pred_path, iou_threshold=0.5):
    """Evaluate predictions and compute Precision, Recall, F1-score per class."""
    with open(gt_path, 'r') as f:
        gt_data = json.load(f)
    with open(pred_path, 'r') as f:
        pred_data = json.load(f)
    
    # Get category information
    categories = {cat['id']: cat['name'] for cat in gt_data['categories']}
    
    # Initialize per-class counters
    class_metrics = {cat_id: {'tp': 0, 'fp': 0, 'fn': 0} for cat_id in categories}
    
    # Organize ground truth and predictions by image
    gt_boxes = {ann['image_id']: [] for ann in gt_data['annotations']}
    for ann in gt_data['annotations']:
        gt_boxes[ann['image_id']].append((ann['bbox'], ann['category_id']))
    
    pred_boxes = {ann['image_id']: [] for ann in pred_data['annotations']}
    for ann in pred_data['annotations']:
        pred_boxes[ann['image_id']].append((ann['bbox'], ann['category_id']))
    
    # Evaluate each image
    for image_id in gt_boxes:
        gt_bboxes = gt_boxes.get(image_id, [])
        pred_bboxes = pred_boxes.get(image_id, [])
        
        matched = set()
        
        # Process predictions
        for pred_bbox, pred_cat in pred_bboxes:
            best_iou = 0
            best_gt_idx = None
            
            for gt_idx, (gt_bbox, gt_cat) in enumerate(gt_bboxes):
                if gt_idx in matched or gt_cat != pred_cat:
                    continue
                
                iou = calc_iou(pred_bbox, gt_bbox)
                if iou > best_iou:
                    best_iou = iou
                    best_gt_idx = gt_idx
            
            if best_iou > iou_threshold and best_gt_idx is not None:
                # True positive for this class
                class_metrics[pred_cat]['tp'] += 1
                matched.add(best_gt_idx)
            else:
                # False positive for this class
                class_metrics[pred_cat]['fp'] += 1
        
        # Count false negatives for each class
        for gt_bbox, gt_cat in gt_bboxes:
            gt_idx = gt_bboxes.index((gt_bbox, gt_cat))
            if gt_idx not in matched:
                class_metrics[gt_cat]['fn'] += 1
    
    # Calculate metrics for each class
    results = {}
    overall_tp, overall_fp, overall_fn = 0, 0, 0
    
    for cat_id, metrics in class_metrics.items():
        tp, fp, fn = metrics['tp'], metrics['fp'], metrics['fn']
        overall_tp += tp
        overall_fp += fp
        overall_fn += fn
        
        precision = tp / (tp + fp) if tp + fp > 0 else 0
        recall = tp / (tp + fn) if tp + fn > 0 else 0
        f1 = 2 * (precision * recall) / (precision + recall) if precision + recall > 0 else 0
        
        results[cat_id] = {
            'category_name': categories[cat_id],
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'support': tp + fn  # Total number of ground truth instances
        }
    
    # Calculate overall metrics
    overall_precision = overall_tp / (overall_tp + overall_fp) if overall_tp + overall_fp > 0 else 0
    overall_recall = overall_tp / (overall_tp + overall_fn) if overall_tp + overall_fn > 0 else 0
    overall_f1 = 2 * (overall_precision * overall_recall) / (overall_precision + overall_recall) if overall_precision + overall_recall > 0 else 0
    
    results['overall'] = {
        'precision': overall_precision,
        'recall': overall_recall,
        'f1_score': overall_f1,
        'support': overall_tp + overall_fn
    }
    
    # Print results
    print(f"{'Category':<20} {'Precision':<10} {'Recall':<10} {'F1 Score':<10} {'Support':<10}")
    print("-" * 80)
    
    for cat_id, metrics in results.items():
        if cat_id == 'overall':
            continue
        name = metrics['category_name']
        print(f"{name:<20} {metrics['precision']:.4f}     {metrics['recall']:.4f}     {metrics['f1_score']:.4f}     {metrics['support']}")
    
    print("-" * 80)
    print(f"{'Overall':<20} {results['overall']['precision']:.4f}     {results['overall']['recall']:.4f}     {results['overall']['f1_score']:.4f}     {results['overall']['support']}")
    
    return results


def compute_confusion_matrix(gt_path, pred_path, iou_threshold=0.5):
    """
    Generate a confusion matrix for object detection results without sklearn.
    
    Args:
        gt_path (str): Path to ground truth JSON file
        pred_path (str): Path to prediction JSON file
        iou_threshold (float): IoU threshold for matching boxes
    
    Returns:
        dict: Confusion matrix as a dictionary with class-to-class predictions
        list: Ordered category names
    """
    with open(gt_path, 'r') as f:
        gt_data = json.load(f)
    with open(pred_path, 'r') as f:
        pred_data = json.load(f)
    
    # Get category information
    categories = {cat['id']: cat['name'] for cat in gt_data['categories']}
    category_ids = sorted(list(categories.keys()))
    category_names = [categories[cat_id] for cat_id in category_ids]
    
    # Initialize confusion matrix with zeros
    # Format: confusion_matrix[true_class][pred_class] = count
    confusion_matrix = {}
    for true_id in category_ids:
        confusion_matrix[true_id] = {}
        for pred_id in category_ids:
            confusion_matrix[true_id][pred_id] = 0
    
    # Add a "missed" category (false negatives)
    missed_category = -1
    for true_id in category_ids:
        confusion_matrix[true_id][missed_category] = 0
    
    # Add a "false alarm" category (false positives without matching class)
    false_alarm_category = -2
    for pred_id in category_ids:
        if false_alarm_category not in confusion_matrix:
            confusion_matrix[false_alarm_category] = {}
        confusion_matrix[false_alarm_category][pred_id] = 0
    
    # Organize ground truth and predictions by image
    gt_boxes = {ann['image_id']: [] for ann in gt_data['annotations']}
    for ann in gt_data['annotations']:
        gt_boxes[ann['image_id']].append((ann['bbox'], ann['category_id']))
    
    pred_boxes = {ann['image_id']: [] for ann in pred_data['annotations']}
    for ann in pred_data['annotations']:
        pred_boxes[ann['image_id']].append((ann['bbox'], ann['category_id']))
    
    # Process each image
    for image_id in gt_boxes:
        gt_bboxes = gt_boxes.get(image_id, [])
        pred_bboxes = pred_boxes.get(image_id, [])
        
        # Track matched ground truth boxes
        matched = set()
        
        # Process predictions
        for pred_bbox, pred_cat in pred_bboxes:
            best_iou = 0
            best_gt_idx = None
            best_gt_cat = None
            
            # Try to match with every ground truth box
            for gt_idx, (gt_bbox, gt_cat) in enumerate(gt_bboxes):
                if gt_idx in matched:
                    continue
                
                iou = calc_iou(pred_bbox, gt_bbox)
                if iou > best_iou:
                    best_iou = iou
                    best_gt_idx = gt_idx
                    best_gt_cat = gt_cat
            
            if best_iou > iou_threshold and best_gt_idx is not None:
                # Matched: increment confusion matrix cell
                confusion_matrix[best_gt_cat][pred_cat] += 1
                matched.add(best_gt_idx)
            else:
                # False positive: increment false alarm counter
                confusion_matrix[false_alarm_category][pred_cat] += 1
        
        # Count unmatched ground truth boxes as misses
        for gt_idx, (gt_bbox, gt_cat) in enumerate(gt_bboxes):
            if gt_idx not in matched:
                confusion_matrix[gt_cat][missed_category] += 1
    
    # Format for printing
    print("\nConfusion Matrix:")
    print("-" * 80)
    
    # Header row with predicted classes
    header = "True\\Pred |"
    for pred_id in category_ids:
        header += f" {categories[pred_id][:7]:7} |"
    header += f" {'Missed':7} |"
    print(header)
    print("-" * 80)
    
    # Print each row of the confusion matrix
    for true_id in category_ids:
        row = f"{categories[true_id][:7]:7} |"
        for pred_id in category_ids:
            row += f" {confusion_matrix[true_id][pred_id]:7d} |"
        row += f" {confusion_matrix[true_id][missed_category]:7d} |"
        print(row)
    
    # Print false positive row
    fp_row = f"{'FalseP':7} |"
    for pred_id in category_ids:
        false_positives = confusion_matrix[false_alarm_category][pred_id]  # IoU-based false positives
        for actual_id in category_ids:
            if actual_id != pred_id:
                false_positives += confusion_matrix[actual_id][pred_id]  # Misclassifications
        fp_row += f" {false_positives:7d} |"
    print("-" * 80)
    print(fp_row)
    
    # Calculate and print performance statistics
    total_correct = sum(confusion_matrix[cat_id][cat_id] for cat_id in category_ids)
    total_gt = sum(sum(confusion_matrix[true_id].values()) for true_id in category_ids)
    total_pred = sum(confusion_matrix[false_alarm_category].values()) + \
                sum(sum(v for k, v in row.items() if k != missed_category) 
                    for true_id, row in confusion_matrix.items() if true_id != false_alarm_category)
    
    accuracy = total_correct / total_gt if total_gt > 0 else 0
    print("-" * 80)
    print(f"Total correct classifications: {total_correct}")
    print(f"Classification accuracy: {accuracy:.4f}")
    
    # Return the confusion matrix and category names for further analysis
    return confusion_matrix, category_names
    

gt_json_path = "/Users/bhavish/Downloads/Ground_Truth_2-at-2025-09-03-06-28-9c56b5fd/result.json"
pred_json_path = "/Users/bhavish/Downloads/Ground_Truth_2-at-2025-09-03-06-28-9c56b5fd/yolo-results_rbc_plt_coco-iter.json"
evaluate_predictions(gt_json_path, pred_json_path)
print("-" * 80)
evaluate_predictions_per_class(gt_json_path, pred_json_path)
print("-" * 80)
compute_confusion_matrix(gt_json_path, pred_json_path)
print("-" * 80)