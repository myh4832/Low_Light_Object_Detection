from ultralytics import YOLO

# Set  model_restoration=True  if pretrained low light enhancement module exists.
# You have to set model_restoration path where the pretrained model positioned. 
# If not, then set  model_restoration=False.

model = YOLO("./runs/detect/train/weights/best.pt", model_restoration=True, model_restoration_path='./experiments/RetinexFormer_COCO/best_psnr_25.68_132000.pth')

# Validation
metrics = model.val()
