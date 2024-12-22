

from ultralytics.models import NAS, RTDETR, SAM, YOLO, FastSAM, YOLOWorld
 
if __name__=="__main__":
 
    # 使用自己的YOLOv11.yamy文件搭建模型并加载预训练权重训练模型
    model = YOLO(r"/content/ultralytics/ultralytics/cfg/models/11/yolo11-HTB.yaml")\
        .load(r'/content/yolo11n.pt')  # build from YAML and transfer weights
 
    results = model.train(data=r'/content/smokier-1/data.yaml',
                          epochs=100, imgsz=640, batch=8)
 
 
 