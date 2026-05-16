import os

import cv2
import torch
import numpy as np
from torchvision import transforms
from PIL import Image
from model import my_model
import utils


def main():
    config = utils.read_config()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 类别顺序必须与训练时的文件夹名称排序一致（按字母序）
    label_name = ['Anger', 'Disgust', 'Fear', 'Happiness', 'Neutral', 'Sadness', 'Surprise']

    # 图像预处理 - 添加归一化，与 ResNet 预训练保持一致
    data_trans = transforms.Compose([
        transforms.Resize(config['img_size'],config['img_size']),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # 实例化模型并加载训练好的权重
    model = my_model(num_classes=7)
    model_path = "./checkpoints/best.pth"

    # 检查模型文件是否存在
    if not os.path.exists(model_path):
        print(f"错误：模型文件不存在: {model_path}")
        print("请先运行 main.py 训练模型")
        return

    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()
    print("模型加载成功，开启摄像头...")

    # 使用 OpenCV 内置的 Haar Cascade 文件路径
    cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'

    if not os.path.exists(cascade_path):
        print(f"错误：找不到 Haar Cascade 文件: {cascade_path}")
        return

    face_cascade = cv2.CascadeClassifier(cascade_path)

    if face_cascade.empty():
        print("错误：Haar Cascade 分类器加载失败")
        return

    print("人脸检测器加载成功")
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("错误：无法打开摄像头")
        return

    print("\n提示：按 'q' 键退出\n")

    frame_count = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            print("无法读取摄像头画面")
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces_rects = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)

        for (x, y, w, h) in faces_rects:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

            # 修正：直接从原始彩色帧提取人脸区域（BGR格式）
            face_bgr = frame[y:y + h, x:x + w]

            # BGR 转 RGB
            face_rgb = cv2.cvtColor(face_bgr, cv2.COLOR_BGR2RGB)

            # 转换为 PIL Image
            face_img = Image.fromarray(face_rgb)

            # 预处理
            face_tensor = data_trans(face_img).unsqueeze(0).to(device)

            # 推理
            with torch.no_grad():
                output = model(face_tensor)
                probabilities = torch.softmax(output, dim=1)  # 获取概率分布
                pred_idx = output.argmax(dim=1).item()
                confidence = probabilities[0][pred_idx].item()
                fer_text = label_name[pred_idx]

            # 显示结果（包含置信度）
            result_text = f"{fer_text} ({confidence:.2f})"
            cv2.putText(frame, result_text, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX,
                        0.8, (0, 0, 255), 2, cv2.LINE_AA)

            # 打印所有类别的概率（用于调试）
            if frame_count % 30 == 0:  # 每30帧打印一次
                print(f"\n检测结果: {fer_text} (置信度: {confidence:.2%})")
                print("各类别概率:")
                for i, (name, prob) in enumerate(zip(label_name, probabilities[0])):
                    bar = "█" * int(prob.item() * 20)
                    print(f"  {name:12s}: {prob.item():.3f} {bar}")

        cv2.imshow('Face Emotion Detection', frame)
        frame_count += 1
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
