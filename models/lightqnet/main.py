from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import cv2
import sys
from time import time, sleep
import numpy as np
import tensorflow.compat.v1 as tf
from tensorflow.python.framework import graph_util
import psutil
import gc
from .common import scan_image_tree
# from tf_face_detection_mtcnn import TfFaceDetectorMtcnn
from .tf_face_quality_model import TfFaceQaulityModel


def read_image(path):
    image_origin = cv2.imdecode(np.fromfile(path, dtype=np.uint8), cv2.IMREAD_UNCHANGED)
    # print(image_origin.shape[0])
    return image_origin


def get_model_info(model):
    """Lấy thông tin về model từ graph"""
    with model.graph.as_default():
        graph_def = model.graph.as_graph_def()
        
        # Đếm số lượng operations
        op_counts = {}
        for node in graph_def.node:
            if node.op not in op_counts:
                op_counts[node.op] = 0
            op_counts[node.op] += 1
            
        # Tính số tham số
        total_params = 0
        for node in graph_def.node:
            if node.op == 'Const':
                try:
                    tensor = tf.make_ndarray(node.attr['value'].tensor)
                    total_params += tensor.size
                except (KeyError, AttributeError):
                    continue
                    
        return op_counts, total_params


def evaluate_model_performance(model, test_image, num_runs=100):
    """Đánh giá hiệu suất model
    
    Args:
        model: Model cần đánh giá
        test_image: Ảnh test
        num_runs: Số lần chạy để tính trung bình
    """
    print("\n=== Model Performance Evaluation ===")
    
    # 1. Thông tin model
    op_counts, total_params = get_model_info(model)
    print("Model Architecture:")
    for op, count in op_counts.items():
        print(f"  - {op}: {count}")
    print(f"Total parameters: {total_params:,}")
    
    # 2. Đánh giá thời gian inference
    times = []
    for _ in range(num_runs):
        start_time = time()
        _ = model.inference(test_image)
        end_time = time()
        times.append((end_time - start_time) * 1000)  # Convert to ms
    
    avg_time = sum(times) / len(times)
    min_time = min(times)
    max_time = max(times)
    print(f"\nInference Performance:")
    print(f"  - Average time: {avg_time:.2f}ms")
    print(f"  - Min time: {min_time:.2f}ms")
    print(f"  - Max time: {max_time:.2f}ms")
    print(f"  - Average FPS: {1000/avg_time:.2f}")
    print(f"  - Max FPS: {1000/min_time:.2f}")
    
    # 3. Đánh giá bộ nhớ
    process = psutil.Process(os.getpid())
    memory_info = process.memory_info()
    print(f"\nMemory Usage:")
    print(f"  - RSS: {memory_info.rss / 1024 / 1024:.2f} MB")
    print(f"  - VMS: {memory_info.vms / 1024 / 1024:.2f} MB")
    
    print("================================\n")


def create_image_grid(images, scores, grid_size=(3, 4)):
    """Tạo grid ảnh với scores
    
    Args:
        images: List các ảnh
        scores: List các score tương ứng
        grid_size: Kích thước grid (rows, cols)
    """
    rows, cols = grid_size
    
    # Định nghĩa kích thước mỗi cell và các padding
    cell_width = 220 # Kích thước chiều rộng của một ô
    cell_height = 280 # Kích thước chiều cao của một ô
    
    outer_grid_padding = 10 # Padding cho toàn bộ grid từ cạnh cửa sổ
    inner_cell_padding_horizontal = 10 # Padding ngang cho ảnh/text trong cell
    inner_cell_padding_vertical_top = 10 # Padding từ đỉnh cell đến ảnh
    score_area_height = 30 # Chiều cao dành cho hiển thị score và padding
    inner_cell_padding_vertical_bottom = 10 # Padding từ score đến đáy cell

    # Tính toán kích thước tổng của grid
    grid_display_width = cell_width * cols + 2 * outer_grid_padding
    grid_display_height = cell_height * rows + 2 * outer_grid_padding

    # Tạo canvas trắng cho grid
    grid = np.ones((grid_display_height, grid_display_width, 3), dtype=np.uint8) * 255
    
    for idx, (img, score) in enumerate(zip(images, scores)):
        if idx >= rows * cols:  # Chỉ hiển thị tối đa rows*cols ảnh
            break

        # Tính vị trí trong grid
        row = idx // cols
        col = idx % cols
            
        # Tính vị trí top-left của cell hiện tại trên grid
        cell_start_x = col * cell_width + outer_grid_padding
        cell_start_y = row * cell_height + outer_grid_padding
        
        # Tính toán không gian dành cho ảnh bên trong cell
        max_image_render_width = cell_width - 2 * inner_cell_padding_horizontal
        max_image_render_height = cell_height - inner_cell_padding_vertical_top - score_area_height - inner_cell_padding_vertical_bottom

        h, w = img.shape[:2]
        aspect_ratio = w / h
        
        # Resize ảnh giữ nguyên tỷ lệ để vừa với không gian dành cho ảnh
        if aspect_ratio > (max_image_render_width / max_image_render_height):
            new_width = max_image_render_width
            new_height = int(new_width / aspect_ratio)
        else:
            new_height = max_image_render_height
            new_width = int(new_height * aspect_ratio)
            
        img_resized = cv2.resize(img, (new_width, new_height))
        
        # Tính toán offset để căn giữa ảnh trong vùng hiển thị ảnh của cell
        img_center_offset_x = (max_image_render_width - new_width) // 2
        img_center_offset_y = (max_image_render_height - new_height) // 2
        
        # Tính toán vị trí paste ảnh vào grid
        paste_img_x = cell_start_x + inner_cell_padding_horizontal + img_center_offset_x
        paste_img_y = cell_start_y + inner_cell_padding_vertical_top + img_center_offset_y

        grid[paste_img_y : paste_img_y + new_height,
             paste_img_x : paste_img_x + new_width] = img_resized
        
        # Thêm score (chỉ hiển thị số)
        score_text = f"{score:.3f}"
        
        # Tính toán vị trí text để không bị đè lên ảnh và có khoảng cách
        # Đặt text ngay dưới vùng ảnh, căn giữa trong vùng score
        text_x_pos = paste_img_x # Căn chỉnh text theo cạnh trái của ảnh
        text_y_pos = paste_img_y + new_height + (score_area_height // 2) + 5 # Đặt text dưới ảnh một khoảng, căn giữa trong vùng score_area_height

        cv2.putText(grid, score_text, 
                   (text_x_pos, text_y_pos),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
    
    return grid


def main(folder_path):
    # Tạo và đánh giá model
    face_qaulity_predictor = TfFaceQaulityModel()
    
    try:
        # Đọc và đánh giá tất cả ảnh
        image_scores = []
        for image_name in os.listdir(folder_path):
            img_path = os.path.join(folder_path, image_name)
            print(f"Processing: {img_path}")
            img = read_image(img_path)
            
            # Chạy inference
            quality_score = face_qaulity_predictor.inference(img)
            image_scores.append((img, quality_score, img_path))
        
        # Sắp xếp theo score từ cao xuống thấp
        image_scores.sort(key=lambda x: x[1], reverse=True)
        
        # Tách images và scores
        images = [x[0] for x in image_scores]
        scores = [x[1] for x in image_scores]
        paths = [x[2] for x in image_scores]
        
        # Tạo grid ảnh
        grid = create_image_grid(images, scores)
        
        # Hiển thị grid
        cv2.imshow('Face Quality Assessment Results', grid)
        print("\nPress any key to exit...")
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        
        # In thông tin chi tiết
        print("\nDetailed Results:")
        for path, score in zip(paths, scores):
            print(f"{os.path.basename(path)}: {score:.4f}")
            
    finally:
        # Cleanup
        face_qaulity_predictor.sess.close()
        tf.reset_default_graph()
        gc.collect()


def run_camera_inference():
    """Chạy inference với camera, tập trung vào vùng giữa chiếm 70% diện tích"""
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Không thể mở camera")
        return

    face_qaulity_predictor = TfFaceQaulityModel()
    
    try:
        # Đánh giá hiệu suất model
        ret, frame = cap.read()
        if ret:
            evaluate_model_performance(face_qaulity_predictor, frame)
        
        print("Nhấn 'q' để thoát")
        
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Không thể đọc frame từ camera")
                break
                
            # Lấy kích thước frame
            height, width = frame.shape[:2]
            
            # Tính toán vùng crop (70% diện tích ở giữa)
            crop_width = int(width * 0.7)
            crop_height = int(height * 0.7)
            
            # Tính toán tọa độ để crop ở giữa
            x1 = (width - crop_width) // 2
            y1 = (height - crop_height) // 2
            x2 = x1 + crop_width
            y2 = y1 + crop_height
            
            # Vẽ khung crop
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            
            # Crop vùng giữa
            roi = frame[y1:y2, x1:x2]
            
            # Đánh giá chất lượng khuôn mặt
            quality_score = face_qaulity_predictor.inference(roi)
            
            # Hiển thị điểm số
            cv2.putText(frame, f"Quality: {quality_score:.4f}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            
            # Hiển thị frame
            cv2.imshow('Face Quality Assessment', frame)
            
            # Thoát khi nhấn 'q'
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    finally:
        # Cleanup
        cap.release()
        cv2.destroyAllWindows()
        face_qaulity_predictor.sess.close()
        tf.reset_default_graph()
        gc.collect()


if __name__ == '__main__':
    # Thêm argument parser để chọn chế độ chạy
    import argparse
    parser = argparse.ArgumentParser(description='Face Quality Assessment')
    parser.add_argument('--camera', type=bool, default=False)
    parser.add_argument('--images', type=str, default='models.lightqnet.test')
    args = parser.parse_args()
    
    if args.camera:
        run_camera_inference()
    else:
        main(args.images)

