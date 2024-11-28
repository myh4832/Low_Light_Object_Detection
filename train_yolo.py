from ultralytics import YOLO
from ultralytics.models.yolo.detect.val import DetectionValidator

model = YOLO("/home/myhong/LOD/LLOD/runs/detect/train3/weights/best.pt", model_restoration=True)

# train_results = model.train(
#     data="/home/myhong/LOD/LLOD/ultralytics/ultralytics/cfg/datasets/coco.yaml",
#     batch=4
# )

metrics = model.val()
