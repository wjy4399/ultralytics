from ultralytics.models import YOLO

# Load a model
model = YOLO(r"D:\code\ultralytics\models\yolo1n-HTB\weights\best.pt")

# Customize validation settings
validation_results = model.val(data="data.yaml", imgsz=640, batch=16, conf=0.25, iou=0.5)

# Load a model
model = YOLO(r"D:\code\ultralytics\models\htb84\train\weights\best.pt")

# Customize validation settings
validation_results = model.val(data="data.yaml", imgsz=640, batch=16, conf=0.25, iou=0.5)

model = YOLO(r"D:\code\ultralytics\models\all\11n848\weights\best.pt")

# Customize validation settings
validation_results = model.val(data="data.yaml", imgsz=640, batch=16, conf=0.25, iou=0.5)