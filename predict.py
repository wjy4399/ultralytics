from ultralytics.models import YOLO

if __name__ == "__main__":
    # 加载训练好的模型
    model = YOLO(r"D:\code\ultralytics\models\all\11n848\weights\best.pt")  # 这里是你训练好的模型权重路径

    # # 推理单张图片
    # results = model.predict(source='/path/to/image.jpg',  # 替换为待预测图片路径
    #                         imgsz=640,  # 推理图片大小
    #                         conf=0.25,  # 置信度阈值
    #                         save=True,  # 是否保存预测结果
    #                         save_txt=True)  # 是否保存结果为文本

    # 推理一个文件夹中的图片
    results = model.predict(source=r"D:\project\毕设\自制数据集\展示",  # 替换为图片文件夹路径
                            imgsz=640,
                            conf=0.25,
                            save=True,
                            save_txt=True)

    # # 推理视频
    # results = model.predict(source='/path/to/video.mp4',  # 替换为待预测视频路径
    #                         imgsz=640,
    #                         conf=0.25,
    #                         save=True)

    # 打印预测结果
    for result in results:
        print(result)  # 可以查看每个目标的位置信息和类别
