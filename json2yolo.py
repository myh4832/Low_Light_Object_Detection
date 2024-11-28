import os
import json
import cv2
from tqdm import tqdm
import argparse

# YOLO 클래스 이름 매핑
NAMES = [
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat", "traffic light",
    "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep",
    "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase",
    "frisbee", "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove", "skateboard",
    "surfboard", "tennis racket", "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl",
    "banana", "apple", "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake",
    "chair", "couch", "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse", "remote",
    "keyboard", "cell phone", "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock",
    "vase", "scissors", "teddy bear", "hair drier", "toothbrush"
]

# NAMES를 역매핑하여 {name: id} 형태로 변환
NAME_TO_ID = {name: idx for idx, name in enumerate(NAMES)}

def read_json(json_file):
    """JSON 파일 읽기"""
    with open(json_file, 'r') as f:
        json_obj = json.load(f)
    return json_obj

def calc_centroids(bbox, img_shape):
    """COCO bbox를 YOLO 포맷으로 변환"""
    xmin, ymin, w, h = bbox
    img_height, img_width = img_shape[:2]
    x_c = xmin + w / 2
    y_c = ymin + h / 2
    x_c_rel = round(x_c / img_width, 6)
    y_c_rel = round(y_c / img_height, 6)
    w_rel = round(w / img_width, 6)
    h_rel = round(h / img_height, 6)
    return (x_c_rel, y_c_rel, w_rel, h_rel)

def write_yolo_txt(rel_bbox, class_id, img_filename, save_dir):
    """YOLO 포맷으로 txt 파일에 데이터 추가"""
    txt_filename = os.path.join(save_dir, os.path.splitext(img_filename)[0] + ".txt")
    with open(txt_filename, "a") as f:  # "a" 모드로 열어 기존 내용에 추가
        row = f"{class_id} {rel_bbox[0]} {rel_bbox[1]} {rel_bbox[2]} {rel_bbox[3]}\n"
        f.write(row)

def convert_to_yolo(json_file, img_dir, save_dir):
    """COCO JSON을 YOLO 포맷으로 변환"""
    # JSON 읽기
    json_obj = read_json(json_file)

    # 이미지 ID와 파일 이름 매핑
    img_to_id_dict = {img['id']: img['file_name'] for img in json_obj['images']}
    cateogries = json_obj["categories"]
    
    # 저장 디렉토리 생성
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    # 전체 annotations 순회
    for anno in tqdm(json_obj['annotations'], desc="Processing annotations"):
        image_id = anno['image_id']  # 이미지 ID
        category_id = anno['category_id']  # COCO category_id
        bbox = anno['bbox']  # COCO bbox: [x_min, y_min, w, h]

        # 이미지 파일 이름 가져오기
        img_name = img_to_id_dict.get(image_id)
        if img_name is None:
            print(f"Warning: Image ID {image_id} not found in images.")
            continue

        img_path = os.path.join(img_dir, img_name)

        # 이미지 파일 확인 및 크기 계산
        if not os.path.exists(img_path):
            print(f"Image {img_path} not found, skipping.")
            continue

        img_shape = cv2.imread(img_path).shape

        # bbox를 YOLO 포맷으로 변환
        rel_bbox = calc_centroids(bbox, img_shape)

        # YOLO 클래스 ID 가져오기
        class_name = [element["name"] for element in cateogries if element["id"] == category_id][0]
        class_id = NAME_TO_ID[class_name]  # names에서 ID 가져오기

        # YOLO 라벨 파일에 데이터 추가
        write_yolo_txt(rel_bbox, class_id, img_name, save_dir)

def main():
    parser = argparse.ArgumentParser(description="Convert COCO JSON annotations to YOLO format.")
    parser.add_argument("--json_file", required=True, help="Path to the COCO JSON annotation file.")
    parser.add_argument("--img_dir", required=True, help="Path to the directory containing images.")
    parser.add_argument("--save_dir", required=True, help="Path to the directory to save YOLO annotations.")
    
    args = parser.parse_args()
    
    convert_to_yolo(args.json_file, args.img_dir, args.save_dir)

if __name__ == "__main__":
    main()