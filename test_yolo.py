import cv2
from ultralytics import YOLO

# 测试 OpenCV
image = cv2.imread("test.jpg")

if image is None:
    raise FileNotFoundError("找不到 test.jpg")

print("OpenCV 读取成功，图片尺寸：", image.shape)

# 加载 YOLO11m 模型
model = YOLO("yolo11m.pt")

# 执行检测
model.predict(
    source="test.jpg",
    save=True,
    conf=0.25,
    device="mps",
    project="outputs",
    name="yolo11m_test"
)

print("YOLO11m 检测完成")