"""
本地模型推理模块（ResNet50 - CUB-200-2011）
===========================================
功能：加载训练好的 PyTorch 模型，对输入图片进行鸟类识别。
返回格式与百度API一致：(bird_name, confidence, description)

该模块被 backend/app.py 调用，用于替换百度API。
"""

import os
import json
import logging
from pathlib import Path

import torch
import torch.nn.functional as F
from torchvision import transforms, models
from PIL import Image

logger = logging.getLogger(__name__)

# 全局变量（模块级缓存，应用启动时加载一次）
_model = None
_class_names = None
_device = None
_model_loaded = False


def _get_default_transform():
    """获取验证/推理用的标准化预处理流程"""
    return transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])


def _build_model(num_classes=200):
    """
    构建与训练时结构一致的 ResNet50 模型。
    注意：必须与 train.py 中的 build_model() 结构完全一致。
    """
    model = models.resnet50(weights=None)  # 不加载预训练权重，后面加载 checkpoint
    in_features = model.fc.in_features
    model.fc = torch.nn.Sequential(
        torch.nn.Dropout(0.2),
        torch.nn.Linear(in_features, 512),
        torch.nn.ReLU(inplace=True),
        torch.nn.BatchNorm1d(512),
        torch.nn.Dropout(0.2),
        torch.nn.Linear(512, num_classes),
    )
    return model


def load_model(model_path=None, class_names_path=None, device=None):
    """
    加载训练好的模型和类别名称。
    
    参数：
        model_path: best_model.pth 路径，默认为 backend/models/best_model.pth
        class_names_path: class_names.json 路径，同上
        device: 运行设备，自动选择 cuda/cpu
        
    返回：
        bool: 是否加载成功
    """
    global _model, _class_names, _device, _model_loaded

    # 自动定位文件路径（相对于本文件所在目录）
    base_dir = Path(__file__).resolve().parent.parent  # backend/
    if model_path is None:
        model_path = base_dir / 'models' / 'best_model.pth'
    if class_names_path is None:
        class_names_path = base_dir / 'models' / 'class_names.json'

    model_path = Path(model_path)
    class_names_path = Path(class_names_path)

    # 检查文件是否存在
    if not model_path.exists():
        logger.warning(f"模型文件不存在: {model_path}")
        _model_loaded = False
        return False
    if not class_names_path.exists():
        logger.warning(f"类别名称文件不存在: {class_names_path}")
        _model_loaded = False
        return False

    # 设备
    if device is None:
        _device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        _device = torch.device(device)
    logger.info(f"推理设备: {_device}")

    # 加载类别名称
    try:
        with open(class_names_path, 'r', encoding='utf-8') as f:
            _class_names = json.load(f)
        # 转换为 {int: name} 格式
        _class_names = {int(k): v for k, v in _class_names.items()}
        logger.info(f"加载了 {len(_class_names)} 个鸟类类别名称")
    except Exception as e:
        logger.error(f"加载类别名称文件失败: {e}")
        _model_loaded = False
        return False

    # 加载模型
    try:
        num_classes = len(_class_names)
        _model = _build_model(num_classes=num_classes)

        # 加载 checkpoint
        checkpoint = torch.load(model_path, map_location=_device)
        if 'model_state_dict' in checkpoint:
            _model.load_state_dict(checkpoint['model_state_dict'])
            if 'val_acc' in checkpoint:
                logger.info(f"模型验证准确率: {checkpoint['val_acc']:.4f}")
        else:
            # 直接是 state_dict
            _model.load_state_dict(checkpoint)

        _model = _model.to(_device)
        _model.eval()
        _model_loaded = True
        logger.info(f"✅ 模型加载成功: {model_path}")
        return True

    except Exception as e:
        logger.error(f"加载模型失败: {e}")
        import traceback
        traceback.print_exc()
        _model_loaded = False
        return False


def predict(image_input, top_k=3):
    """
    对输入的图片进行鸟类识别。
    
    参数：
        image_input: 可以是以下类型之一
            - PIL Image 对象
            - 图片文件路径 (str 或 Path)
            - 图片二进制字节数据 (bytes)
        top_k: 返回前 top_k 个预测结果
        
    返回：
        (bird_name, confidence, description) 成功时
        (None, None, error_message) 失败时
    """
    global _model, _class_names, _device, _model_loaded

    if not _model_loaded or _model is None:
        msg = "模型未加载，请先调用 load_model() 或确认模型文件存在"
        logger.error(msg)
        return None, None, msg

    # 1. 将输入转为 PIL Image
    try:
        if isinstance(image_input, (str, Path)):
            # 文件路径
            image = Image.open(image_input).convert('RGB')
        elif isinstance(image_input, bytes):
            # 二进制数据
            from io import BytesIO
            image = Image.open(BytesIO(image_input)).convert('RGB')
        elif isinstance(image_input, Image.Image):
            image = image_input.convert('RGB')
        else:
            return None, None, f"不支持的输入类型: {type(image_input)}"
    except Exception as e:
        return None, None, f"图片读取失败: {str(e)}"

    # 2. 预处理
    try:
        transform = _get_default_transform()
        input_tensor = transform(image).unsqueeze(0).to(_device)
    except Exception as e:
        return None, None, f"图片预处理失败: {str(e)}"

    # 3. 推理
    try:
        with torch.no_grad():
            outputs = _model(input_tensor)
            probabilities = F.softmax(outputs, dim=1)[0]
            top_probs, top_indices = torch.topk(probabilities, k=min(top_k, len(_class_names)))
    except Exception as e:
        return None, None, f"模型推理失败: {str(e)}"

    # 4. 解析结果
    top1_idx = top_indices[0].item()
    top1_prob = top_probs[0].item()
    bird_name = _class_names.get(top1_idx, f"未知类别({top1_idx})")

    # 构建描述信息（含top-3结果）
    details = []
    for i in range(min(top_k, len(top_indices))):
        idx = top_indices[i].item()
        prob = top_probs[i].item()
        name = _class_names.get(idx, f"未知({idx})")
        details.append(f"{name}({prob:.1%})")

    description = f"识别为{bird_name}，置信度{top1_prob:.1%}。"
    if top_k > 1:
        description += f" 其他可能: {'; '.join(details[1:])}"

    logger.info(f"推理完成: {bird_name} ({top1_prob:.4f})")
    return bird_name, top1_prob, description


def is_model_loaded():
    """查询模型是否已加载"""
    return _model_loaded


# 直接运行本模块时，进行快速测试
if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)

    # 加载模型
    success = load_model()
    if success:
        # 测试预测（使用一张示例图片）
        test_img = Path(__file__).resolve().parent.parent / 'uploads'
        if test_img.exists() and any(test_img.iterdir()):
            for img_file in test_img.iterdir():
                if img_file.suffix.lower() in ('.jpg', '.jpeg', '.png', '.webp'):
                    print(f"测试图片: {img_file}")
                    name, conf, desc = predict(img_file)
                    print(f"  结果: {name} ({conf:.4f})")
                    print(f"  描述: {desc}")
                    break
        else:
            print("没有测试图片，请在 uploads/ 目录放一张图片")
    else:
        print("模型加载失败，请先训练模型")
