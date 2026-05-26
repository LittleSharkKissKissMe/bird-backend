"""
鸟类识别与问答系统 - Flask 后端
==============================
功能：
  1. POST /api/recognize - 接收图片，使用本地模型（CUB-200-2011 微调 ResNet50）识别鸟类

  2. POST /api/ask       - 接收鸟类问题和鸟名，通过知识库匹配或DeepSeek API返回答案
  3. GET  /api/history   - 返回最近识别记录

运行方式：
  pip install -r requirements.txt
  python app.py

需配置环境变量（见 .env.example）：

  DEEPSEEK_API_KEY
"""

import os
import json
import traceback
import logging
import uuid
from datetime import datetime
from pathlib import Path

from flask import Flask, request, jsonify
from flask_cors import CORS
from dotenv import load_dotenv
import requests

from PIL import Image
import torch
import torch.nn.functional as F
from torchvision import transforms, models

# ---------- 模块路径 ----------
# 确保 Python 能找到 recognition/ 和 knowledge/ 模块
_BASE_DIR = Path(__file__).resolve().parent
import sys
sys.path.insert(0, str(_BASE_DIR))

# 项目根目录（backend 的父目录）
_PROJECT_ROOT = _BASE_DIR.parent

from knowledge.bird_knowledge import KNOWLEDGE, KEYWORD_MAP

# ---------- YOLO 模型路径 ----------
# 模型文件统一存放在项目根目录的 models/yolo/ 下
YOLO_MODEL_DIR = _PROJECT_ROOT / "models" / "yolo"
YOLO_MODEL_PATH = YOLO_MODEL_DIR / "yolov8n.pt"

# ---------- YOLO 模型全局句柄 ----------
_yolo_model = None

import base64
import io




# ---------- 加载环境变量 ----------
load_dotenv()

# ---------- 日志配置 ----------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

# ---------- Flask 应用初始化 ----------
app = Flask(__name__)
CORS(app)  # 允许跨域请求

# ---------- 配置 ----------
UPLOAD_FOLDER = _BASE_DIR / "uploads"
UPLOAD_FOLDER.mkdir(exist_ok=True)







# DeepSeek 配置
DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY", "")
DEEPSEEK_API_URL = "https://api.deepseek.com/v1/chat/completions"

# ---------- auto_info 缓存 ----------
# 内存缓存：{bird_name: auto_info_text}
# 同一鸟名只调用一次 DeepSeek API，后续直接返回缓存
_auto_info_cache = {}

# ---------- SQLite 数据库配置 ----------
import sqlite3

DB_PATH = _BASE_DIR / "bird_records.db"


def init_db():
    """初始化SQLite数据库，建表（如果不存在）"""
    conn = sqlite3.connect(str(DB_PATH))
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS recognition_history (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT NOT NULL,
            image_filename TEXT,
            bird_name TEXT,
            confidence REAL
        )
    """)
    conn.commit()
    conn.close()
    logger.info("数据库初始化完成")


def save_history(image_filename, bird_name, confidence):
    """保存识别记录到SQLite"""
    try:
        conn = sqlite3.connect(str(DB_PATH))
        cursor = conn.cursor()
        cursor.execute("""
            INSERT INTO recognition_history (timestamp, image_filename, bird_name, confidence)
            VALUES (?, ?, ?, ?)
        """, (datetime.now().isoformat(), image_filename, bird_name, confidence))
        conn.commit()
        conn.close()
    except Exception as e:
        logger.error(f"保存历史记录失败: {e}")


def get_history(limit=10):
    """获取最近limit条识别记录"""
    try:
        conn = sqlite3.connect(str(DB_PATH))
        cursor = conn.cursor()
        cursor.execute("""
            SELECT timestamp, bird_name, confidence
            FROM recognition_history
            ORDER BY id DESC
            LIMIT ?
        """, (limit,))
        rows = cursor.fetchall()
        conn.close()
        return [
            {"timestamp": row[0], "bird_name": row[1], "confidence": row[2]}
            for row in rows
        ]
    except Exception as e:
        logger.error(f"读取历史记录失败: {e}")
        return []


# ======================================================================
#  YOLO 鸟类轮廓检测模块
# ======================================================================

def load_yolo_model():
    """
    加载 YOLOv8 模型（用于鸟类目标检测）。
    模型文件存放在项目根目录 models/yolo/yolov8n.pt。
    如果文件不存在，自动下载到该路径。
    加载失败时不影响主流程，只是 has_bird 返回 False。
    """
    global _yolo_model

    try:
        from ultralytics import YOLO
    except ImportError:
        logger.warning("ultralytics 未安装，请执行: pip install ultralytics")
        _yolo_model = None
        return False

    # 确保模型目录存在
    YOLO_MODEL_DIR.mkdir(parents=True, exist_ok=True)

    if not YOLO_MODEL_PATH.exists():
        logger.info(f"YOLO 模型文件不存在，正在下载到: {YOLO_MODEL_PATH}")
        try:
            # 使用 ultralytics 的下载功能，指定保存路径
            from ultralytics.utils import download
            url = "https://github.com/ultralytics/assets/releases/download/v8.2.0/yolov8n.pt"
            download(url, str(YOLO_MODEL_PATH))
            logger.info(f"YOLO 模型下载完成")
        except Exception as e:
            logger.error(f"YOLO 模型下载失败: {e}，尝试使用默认缓存路径")
            # 如果下载失败，让 YOLO 从默认缓存加载（会自动下载）
            try:
                _yolo_model = YOLO("yolov8n.pt")
                logger.info("YOLO 模型从缓存加载成功")
                return True
            except Exception as e2:
                logger.error(f"YOLO 模型加载失败（缓存也失败）: {e2}")
                _yolo_model = None
                return False

    try:
        _yolo_model = YOLO(str(YOLO_MODEL_PATH))
        logger.info(f"✅ YOLO 模型加载成功: {YOLO_MODEL_PATH}")
        return True
    except Exception as e:
        logger.error(f"YOLO 模型加载失败: {e}")
        _yolo_model = None
        return False


def detect_bird(image_bytes):
    """
    使用 YOLOv8 检测图片中的鸟类。
    
    参数：
        image_bytes: 原始图片二进制数据（bytes）
    
    返回：
        (has_bird, boxes, annotated_image_base64)
        has_bird: bool - 是否检测到鸟类
        boxes: list[list[int]] - 每个框的 [x1, y1, x2, y2] 整数像素坐标
        annotated_image_base64: str - 画好框的 JPEG 图片 Base64 字符串，
                                     格式 "data:image/jpeg;base64,xxx"
                                     未检测到鸟时返回空字符串 ""
    """
    if _yolo_model is None:
        logger.warning("YOLO 模型未加载，跳过目标检测")
        return False, [], ""

    try:
        # 用 PIL 读取图片，然后转为 numpy 数组供 YOLO 使用
        from PIL import ImageDraw, Image

        img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        img_np = img.copy()  # 保留原图用于画框

        # 推理（只检测 bird，COCO 类别 id=14）
        results = _yolo_model(img, conf=0.5, classes=[14], verbose=False)

        boxes = []
        has_bird = False

        if results and len(results) > 0:
            result = results[0]
            if result.boxes is not None and len(result.boxes) > 0:
                has_bird = True
                # 获取边界框坐标（xyxy 格式）
                xyxy_list = result.boxes.xyxy.cpu().numpy().tolist()
                for box in xyxy_list:
                    x1, y1, x2, y2 = [int(round(v)) for v in box]
                    boxes.append([x1, y1, x2, y2])

                # 在原图上画框
                draw = ImageDraw.Draw(img_np)
                for x1, y1, x2, y2 in boxes:
                    # 绿色框，线宽 3
                    draw.rectangle([x1, y1, x2, y2], outline=(0, 255, 0), width=3)

        if not has_bird:
            return False, [], ""

        # 将画好框的图片编码为 Base64（JPEG 格式）
        buffer = io.BytesIO()
        img_np.save(buffer, format="JPEG", quality=90)
        b64_str = base64.b64encode(buffer.getvalue()).decode("utf-8")
        annotated_image_base64 = f"data:image/jpeg;base64,{b64_str}"

        logger.info(f"YOLO 检测到 {len(boxes)} 只鸟")
        return True, boxes, annotated_image_base64

    except Exception as e:
        logger.error(f"YOLO 目标检测异常（降级处理）: {e}")
        import traceback
        traceback.print_exc()
        return False, [], ""


# ======================================================================

#  本地模型加载与推理模块（自包含，不依赖 recognition/infer.py）
# ======================================================================


























# 全局推理模块状态
_local_model = None
_local_class_names = None          # {int: str}
_local_device = None
_local_model_loaded = False



def _build_resnet50_model(num_classes=200):
    """构建与训练时结构一致的 ResNet50 模型。"""
    model = models.resnet50(weights=None)
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


def _get_preprocess_transform():
    """获取与训练时完全一致的预处理流程。"""
    return transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])


def load_local_model():
    """

    加载本地训练好的 PyTorch 模型和类别名称映射。
    



    使用绝对路径定位模型文件：
      backend/models/best_model.pth
      backend/models/class_names.json
    
    返回：


        bool: 是否加载成功
    """



    global _local_model, _local_class_names, _local_device, _local_model_loaded



    base_dir = os.path.dirname(os.path.abspath(__file__))  # backend/
    model_path = os.path.join(base_dir, 'models', 'best_model.pth')
    class_names_path = os.path.join(base_dir, 'models', 'class_names.json')





    # 检查文件是否存在
    if not os.path.isfile(model_path):
        logger.error(f"模型文件不存在: {model_path}")
        _local_model_loaded = False
        return False
    if not os.path.isfile(class_names_path):
        logger.error(f"类别名称文件不存在: {class_names_path}")
        _local_model_loaded = False
        return False

    # 自动选择设备
    _local_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"推理设备: {_local_device}")

    # 加载类别名称映射
    try:




        with open(class_names_path, 'r', encoding='utf-8') as f:
            raw = json.load(f)
        _local_class_names = {int(k): v for k, v in raw.items()}
        logger.info(f"加载了 {len(_local_class_names)} 个鸟类类别名称")
    except Exception as e:
        logger.error(f"加载类别名称文件失败: {e}")
        traceback.print_exc()
        _local_model_loaded = False
        return False

    # 构建模型并加载权重
    try:






        num_classes = len(_local_class_names)
        _local_model = _build_resnet50_model(num_classes=num_classes)







        checkpoint = torch.load(model_path, map_location=_local_device)
        if 'model_state_dict' in checkpoint:
            _local_model.load_state_dict(checkpoint['model_state_dict'])
            if 'val_acc' in checkpoint:
                logger.info(f"模型验证准确率: {checkpoint['val_acc']:.4f}")
        else:
            _local_model.load_state_dict(checkpoint)


        _local_model = _local_model.to(_local_device)
        _local_model.eval()
        _local_model_loaded = True
        logger.info(f"✅ 本地模型加载成功: {model_path}")
        return True

    except Exception as e:
        logger.error(f"加载模型失败: {e}")
        traceback.print_exc()
        _local_model_loaded = False
        return False












def is_model_loaded():
    """查询本地模型是否已加载。"""
    return _local_model_loaded
































def predict_top_k(image_bytes, top_k=3):
    """

    使用本地模型推理图片，返回概率最高的前 K 个结果。
    
    参数：


        image_bytes: 图片二进制数据（bytes）
        top_k:      返回前 k 个结果，默认 3
    
    返回：




        [{"bird_name": str, "confidence": float}, ...]
    
    异常：
        模型未加载或推理失败时抛出 RuntimeError，附带详细错误信息。
    """


    if not _local_model_loaded or _local_model is None:
        raise RuntimeError("本地模型未加载，请先调用 load_local_model() 或确认模型文件存在。")


    # 1. 读取图片
    try:
        from io import BytesIO
        image = Image.open(BytesIO(image_bytes)).convert('RGB')
    except Exception as e:
        raise RuntimeError(f"图片读取失败: {e}") from e



    # 2. 预处理（与训练时完全一致）
    try:
        transform = _get_preprocess_transform()
        input_tensor = transform(image).unsqueeze(0).to(_local_device)
    except Exception as e:
        raise RuntimeError(f"图片预处理失败: {e}") from e











    # 3. 推理
    try:


        with torch.no_grad():
            outputs = _local_model(input_tensor)
            probabilities = F.softmax(outputs, dim=1)[0]
            top_probs, top_indices = torch.topk(
                probabilities, k=min(top_k, len(_local_class_names))
            )
    except Exception as e:
        raise RuntimeError(f"模型推理失败: {e}") from e


    # 4. 解析结果
    results = []
    for i in range(top_indices.size(0)):
        idx = top_indices[i].item()
        prob = top_probs[i].item()
        name = _local_class_names.get(idx, f"未知类别({idx})")
        results.append({"bird_name": name, "confidence": round(prob, 4)})




    return results





































def recognize_bird(image_bytes):
    """

    纯本地模型识别鸟类（无任何第三方 API 回退）。
    返回 top-3 候选结果。
    
    参数：
        image_bytes: 图片二进制数据
    
    返回：



        (bird_name, confidence, description, top3_list)
        top3_list: [{"bird_name": str, "confidence": float}, ...] 最多3个
    
    异常：
        当模型未加载或推理失败时，抛出 RuntimeError。
    """
    logger.info("使用本地模型识别...")
    top3_list = predict_top_k(image_bytes, top_k=3)




    name = top3_list[0]["bird_name"]
    conf = top3_list[0]["confidence"]

    desc = f"识别为{name}，置信度{conf:.1%}。"
    if len(top3_list) > 1:
        others = "; ".join(
            [f"{t['bird_name']}({t['confidence']:.1%})" for t in top3_list[1:]]
        )
        desc += f" 其他可能: {others}"

    # 如果知识库中有该鸟类的趣味知识，附加到描述中
    if name in KNOWLEDGE:
        extra = KNOWLEDGE[name].get("趣味知识", "")
        if extra:
            desc = desc + " " + extra

    return name, conf, desc, top3_list





























# ======================================================================
#  auto_info 生成模块（识别返回的科普信息）
# ======================================================================

def generate_auto_info(bird_name):
    """
    根据识别的鸟名生成一段科普简介（auto_info）。
    
    策略：
      1. 检查内存缓存，命中则直接返回
      2. 如果鸟名在本地 KNOWLEDGE 中，从知识库字段格式化拼接
      3. 如果不在本地知识库，调用 DeepSeek API 生成（≤150字），并缓存
    
    参数：
        bird_name: 鸟类名称（字符串）
    
    返回：
        auto_info 文本字符串，或 None（生成失败时）
    """
    global _auto_info_cache

    # 1. 缓存命中
    if bird_name in _auto_info_cache:
        logger.info(f"[auto_info] 缓存命中: {bird_name}")
        return _auto_info_cache[bird_name]

    # 2. 本地知识库
    if bird_name in KNOWLEDGE:
        info = KNOWLEDGE[bird_name]
        parts = []

        # 外形特征（如果有）
        if "外形特征" in info:
            parts.append(info["外形特征"].rstrip("。"))
        # 食性
        if "食性" in info:
            parts.append(info["食性"].rstrip("。"))
        # 分布
        if "分布" in info:
            parts.append(info["分布"].rstrip("。"))
        # 保护等级
        if "保护等级" in info:
            parts.append(info["保护等级"].rstrip("。"))
        # 趣味知识（可选一段有趣的）
        if "趣味知识" in info:
            # 取前半句作为补充
            fun = info["趣味知识"].rstrip("。")
            if len(fun) > 30:
                # 取第一句（如果有句号）
                first_sentence = fun.split("。")[0] if "。" in fun else fun[:30]
                parts.append(first_sentence)
            else:
                parts.append(fun)

        if parts:
            auto_info = "。".join(parts) + "。"
            _auto_info_cache[bird_name] = auto_info
            logger.info(f"[auto_info] 本地知识库生成成功: {bird_name}")
            return auto_info

    # 3. DeepSeek API 生成（鸟名不在本地知识库中）
    logger.info(f"[auto_info] 本地知识库未收录 {bird_name}，调用 DeepSeek 生成...")
    auto_info = _generate_auto_info_via_deepseek(bird_name)
    if auto_info:
        _auto_info_cache[bird_name] = auto_info
        logger.info(f"[auto_info] DeepSeek 生成成功: {bird_name}")
        return auto_info

    # 4. 全部失败
    logger.warning(f"[auto_info] 生成失败: {bird_name}")
    return None


def _generate_auto_info_via_deepseek(bird_name):
    """
    调用 DeepSeek API 为未见过的鸟类生成一段简短的科普介绍（≤150字）。
    
    参数：
        bird_name: 鸟类名称
    
    返回：
        生成的文本，或 None（失败时）
    """
    if not DEEPSEEK_API_KEY:
        logger.warning("[auto_info] DeepSeek API Key 未配置，无法生成")
        return None

    system_prompt = (
        "你是一个鸟类学家。请用一句或两句话（不超过150字）简要介绍这种鸟类，"
        "内容包括：基本特征、分布区域、食性、保护状况。语言简洁流畅。"
    )
    user_prompt = f"请简要介绍「{bird_name}」这种鸟类。"

    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {DEEPSEEK_API_KEY}",
    }

    payload = {
        "model": "deepseek-chat",
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        "temperature": 0.6,
        "max_tokens": 150,
        "stream": False,
    }

    try:
        resp = requests.post(
            DEEPSEEK_API_URL,
            headers=headers,
            json=payload,
            timeout=20,
        )
        data = resp.json()

        if "choices" in data and len(data["choices"]) > 0:
            answer = data["choices"][0]["message"]["content"].strip()
            # 限制长度不超过 200 字（含标点）
            if len(answer) > 200:
                answer = answer[:200]
            return answer
        else:
            error_info = data.get("error", {}).get("message", "未知错误")
            logger.error(f"[auto_info] DeepSeek API 异常: {error_info}")
            return None

    except requests.Timeout:
        logger.error("[auto_info] DeepSeek API 请求超时")
        return None
    except requests.RequestException as e:
        logger.error(f"[auto_info] DeepSeek API 请求异常: {e}")
        return None


# ======================================================================
#  Location 位置辅助模块（基于 DeepSeek API）
# ======================================================================

def call_deepseek_for_location(location, top3_list):
    """
    基于用户所在位置和视觉识别候选列表，调用 DeepSeek API
    给出该地区最可能的一种鸟类及理由。
    
    参数：
        location: 用户所在位置（如"浙江省杭州市"）
        top3_list: 视觉模型 top-3 候选列表，
                   [{"bird_name": str, "confidence": float}, ...]
    
    返回：
        str: DeepSeek 返回的文本，失败时返回空字符串 ""
    
    说明：
        - 从环境变量 DEEPSEEK_API_KEY 读取 API key
        - 超时 3 秒，超时或异常时返回空字符串
        - 不抛出异常
    """
    if not DEEPSEEK_API_KEY:
        logger.warning("[location] DeepSeek API Key 未配置，无法生成位置建议")
        return ""

    # 构建 top-k 列表的文本描述
    top_k_text = "、".join(
        [f"{item['bird_name']}(置信度{item['confidence']:.1%})" for item in top3_list]
    )

    # 系统提示词
    system_prompt = (
        "你是一位鸟类学家。请根据用户所在的地区，以及视觉模型识别出的候选鸟类列表，"
        "给出该地区最可能的一种鸟。简要说明理由。"
    )

    # 用户提示词
    user_prompt = (
        f"用户位于{location}，当前视觉模型识别可能为以下鸟类：{top_k_text}。"
        f"请根据该地区的常见鸟类，给出最可能的一种鸟名，并简要说明理由。"
    )

    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {DEEPSEEK_API_KEY}",
    }

    payload = {
        "model": "deepseek-chat",
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        "temperature": 0.3,
        "max_tokens": 300,
        "stream": False,
    }

    try:
        resp = requests.post(
            DEEPSEEK_API_URL,
            headers=headers,
            json=payload,
            timeout=15,  # 15秒超时，与 auto_info 一致
        )
        data = resp.json()

        if "choices" in data and len(data["choices"]) > 0:
            answer = data["choices"][0]["message"]["content"].strip()
            logger.info(f"[location] DeepSeek 返回成功: {answer[:60]}...")
            return answer
        else:
            error_info = data.get("error", {}).get("message", "未知错误")
            logger.error(f"[location] DeepSeek API 异常: {error_info}")
            return ""

    except requests.Timeout:
        logger.error("[location] DeepSeek API 请求超时（3s）")
        return ""
    except requests.RequestException as e:
        logger.error(f"[location] DeepSeek API 请求异常: {e}")
        return ""


# ======================================================================
#  智能问答模块
# ======================================================================

# 中英文鸟类名称映射表（中文名 → 英文名, 学名）
# 用于在提示词中提供更丰富的上下文
BIRD_NAME_MAP = {
    # 本地知识库已有的 12 种鸟
    "麻雀":          ("Eurasian Tree Sparrow",        "Passer montanus"),
    "鸽子":          ("Rock Dove / Pigeon",           "Columba livia"),
    "乌鸦":          ("Carrion Crow",                 "Corvus corone"),
    "喜鹊":          ("Eurasian Magpie",              "Pica pica"),
    "翠鸟":          ("Common Kingfisher",            "Alcedo atthis"),
    "啄木鸟":        ("Great Spotted Woodpecker",     "Dendrocopos major"),
    "鹦鹉":          ("Parrot (general)",             "Psittaciformes"),
    "猫头鹰":        ("Eagle Owl / Tawny Owl",        "Strigiformes"),
    "燕子":          ("Barn Swallow",                 "Hirundo rustica"),
    "白鹭":          ("Little Egret",                 "Egretta garzetta"),
    "黄鹂":          ("Black-naped Oriole",           "Oriolus chinensis"),
    "布谷鸟（大杜鹃）": ("Common Cuckoo",              "Cuculus canorus"),
    # 常见的 CUB-200-2011 测试结果（英文名 → 中文/学名）
    "Black_footed_Albatross":  ("Black-footed Albatross",  "Phoebastria nigripes"),
    "Sooty_Albatross":         ("Sooty Albatross",         "Phoebetria fusca"),
    "Northern_Fulmar":         ("Northern Fulmar",         "Fulmarus glacialis"),
}


def _get_bird_name_context(bird_name):
    """
    获取鸟类的多语言名称和知识库上下文。
    
    返回：
        (chinese_name, english_name, scientific_name, knowledge_text)
        各项可能为 None
    """
    chinese_name = bird_name
    english_name = None
    scientific_name = None

    # 从映射表查找
    if bird_name in BIRD_NAME_MAP:
        english_name, scientific_name = BIRD_NAME_MAP[bird_name]
    else:
        # 如果鸟名本身是英文（如 Black_footed_Albatross），
        # 则当作英文名，中文名留空让模型自己判断
        chinese_name = None

    # 获取知识库数据
    knowledge_text = None
    if bird_name in KNOWLEDGE:
        info = KNOWLEDGE[bird_name]
        lines = []
        for key, value in info.items():
            lines.append(f"- {key}：{value}")
        knowledge_text = "\n".join(lines)

    return chinese_name, english_name, scientific_name, knowledge_text


def match_knowledge(bird_name, question):
    """
    根据关键词从知识库中匹配答案。
    
    参数：
        bird_name: 鸟名
        question: 用户问题
        
    返回：
        匹配到的答案，或 None（表示未匹配）
    """
    if bird_name not in KNOWLEDGE:
        return None

    bird_info = KNOWLEDGE[bird_name]

    # 遍历关键词映射表，查找匹配项
    for keyword, field in KEYWORD_MAP.items():
        if keyword in question:
            if field in bird_info:
                return bird_info[field]

    return None


def ask_deepseek(bird_name, question):
    """
    调用 DeepSeek 大模型 API 生成鸟类专家级回答。
    
    参数：
        bird_name: 鸟名
        question: 用户问题
        
    返回：
        生成的答案文本，或 None（调用失败时）
    """
    if not DEEPSEEK_API_KEY:
        logger.warning("[问答] DeepSeek API Key 未配置")
        return None

    # --- 1. 构建丰富的鸟类上下文 ---
    chinese_name, english_name, scientific_name, knowledge_text = \
        _get_bird_name_context(bird_name)

    # 构建名称描述
    name_parts = []
    if chinese_name:
        name_parts.append(f"中文名：{chinese_name}")
    if english_name:
        name_parts.append(f"英文名：{english_name}")
    if scientific_name:
        name_parts.append(f"学名：{scientific_name}")

    # 如果没有中文名（识别结果是英文），让模型根据自身知识判断
    if not chinese_name and english_name:
        name_parts.append(f"（该鸟类的确切中文名请根据你的知识判断）")

    name_context = "、".join(name_parts) if name_parts else bird_name

    # 构建知识库上下文
    if knowledge_text:
        context_block = (
            f"以下是系统中已有的关于该鸟类的参考信息（请以此为基础，结合你自身的专业知识回答）：\n"
            f"{knowledge_text}"
        )
    else:
        context_block = (
            f"系统中没有该鸟类的预置信息。请根据你自身的鸟类学专业知识来回答，"
            f"确保信息的准确性。"
        )

    # --- 2. 构建 System Prompt（专业鸟类学家设定） ---
    system_prompt = (
        "你是一位资深的鸟类学专家，知识渊博且热爱科普。"
        "请用专业、清晰、热情的方式回答用户关于鸟类的问题。"
        "回答时尽量包含饮食、分布、保护现状、趣味知识等要点。"
        "如果不知道，请坦诚相告。"
        "只讨论鸟类相关话题。"
        "使用中文。"
    )

    # --- 3. 构建 User Prompt（包含完整上下文） ---
    user_prompt = (
        f"【鸟类信息】\n"
        f"用户询问的鸟类：{name_context}\n\n"
        f"{context_block}\n\n"
        f"【用户问题】\n"
        f"{question}\n\n"
        f"请回答："
    )

    # --- 4. 调用 DeepSeek API ---
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {DEEPSEEK_API_KEY}",
    }

    payload = {
        "model": "deepseek-chat",
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        "temperature": 0.1,       # 低温度 → 更确定、更事实性的回答
        "max_tokens": 500,        # 限制回答长度
        "stream": False,
    }

    logger.info(f"[问答] 调用 DeepSeek - 鸟类: {bird_name}, 问题: {question[:40]}...")

    try:
        resp = requests.post(
            DEEPSEEK_API_URL,
            headers=headers,
            json=payload,
            timeout=30,
        )
        data = resp.json()

        if "choices" in data and len(data["choices"]) > 0:
            answer = data["choices"][0]["message"]["content"].strip()
            logger.info(f"[问答] DeepSeek 回答成功: {answer[:60]}...")
            return answer
        else:
            error_info = data.get("error", {}).get("message", "未知错误")
            logger.error(f"[问答] DeepSeek API 返回异常: {error_info}")
            return None

    except requests.Timeout:
        logger.error("[问答] DeepSeek API 请求超时")
        return None
    except requests.RequestException as e:
        logger.error(f"[问答] DeepSeek API 请求异常: {e}")
        return None


# ======================================================================
#  API 路由
# ======================================================================

@app.route("/api/recognize", methods=["POST"])
def api_recognize():
    """
    图片识别接口
    --------
    接收前端上传的图片文件（字段名: image），


    使用本地 ResNet50 模型（CUB-200-2011 微调）进行鸟类识别。
    只依赖本地模型，无任何第三方 API 回退。
    
    请求参数（multipart/form-data）：

        image:    图片文件（必需）
        location: 用户所在位置，如"浙江省杭州市"（可选字符串）
    
    成功返回：

        {
            "success": true,
            "bird_name": "麻雀",
            "confidence": 0.96,
            "top3": [{"bird_name": "麻雀", "confidence": 0.96}, ...],
            "description": "...",
            "auto_info": "...",

            "location_suggestion": "..."   // 提供location时有值，否则null
        }
    失败返回：
        { "success": false, "error": "错误信息" }
    """
    # 1. 检查是否有文件上传
    if "image" not in request.files:
        return jsonify({"success": False, "error": "请上传图片文件"}), 400

    file = request.files["image"]

    if file.filename == "":
        return jsonify({"success": False, "error": "请选择一个图片文件"}), 400


    # 2. 读取可选字段 location
    location = request.form.get("location", "").strip()
    if not location:
        location = None

    # 3. 读取图片数据
    try:
        image_bytes = file.read()
    except Exception as e:
        logger.error(f"读取上传文件失败: {e}")
        return jsonify({"success": False, "error": "读取图片失败，请重试"}), 500


    # 4. 检查图片大小（限制10MB）
    if len(image_bytes) > 10 * 1024 * 1024:
        return jsonify({"success": False, "error": "图片过大，请上传小于10MB的图片"}), 400


    # 5. 保存图片（用于记录）
    original_filename = file.filename or f"bird_{uuid.uuid4().hex[:8]}.jpg"
    save_path = UPLOAD_FOLDER / f"{uuid.uuid4().hex}_{original_filename}"
    try:
        save_path.write_bytes(image_bytes)
    except Exception as e:
        logger.warning(f"保存图片文件失败（不影响识别）: {e}")
        save_path = None


    # 6. 调用 YOLO 检测鸟类轮廓（降级不影响主流程）
    has_bird, boxes, annotated_image = detect_bird(image_bytes)

    # 7. 调用本地模型分类识别
    #    失败时直接抛出异常，让统一错误处理捕获
    try:
        bird_name, confidence, description, top3_list = recognize_bird(image_bytes)
    except RuntimeError as e:
        logger.error(f"本地模型识别失败: {e}")
        traceback.print_exc()
        return jsonify({
            "success": False,
            "error": f"识别失败: {str(e)}",
        }), 500








    # 8. 保存识别记录到数据库
    if bird_name and save_path:
        save_history(save_path.name, bird_name, confidence or 0)



    # 9. 生成 auto_info 科普信息
    auto_info = generate_auto_info(bird_name) if bird_name else None



    # 10. 根据 location 生成 location_suggestion
    location_suggestion = None
    if location and top3_list and bird_name:
        logger.info(f"用户提供了位置: {location}，开始调用 DeepSeek 生成位置建议...")
        location_suggestion = call_deepseek_for_location(location, top3_list)
        if not location_suggestion:
            location_suggestion = None  # 失败时返回 null


    # 11. 返回结果
    if bird_name:

        logger.info(f"识别成功: {bird_name} ({confidence:.4f})")
        result = {
            "success": True,
            "bird_name": bird_name,
            "confidence": round(confidence, 4) if confidence else 0,
            "top3": top3_list or [],
            "description": description,

            "has_bird": has_bird,
            "annotated_image": annotated_image if has_bird else "",
            "location_suggestion": location_suggestion,
        }
        if auto_info:
            result["auto_info"] = auto_info
        return jsonify(result)
    else:
        return jsonify({
            "success": False,
            "error": description or "未识别出鸟类，请尝试更清晰的图片",
        })


@app.route("/api/ask", methods=["POST"])
def api_ask():
    """
    智能问答接口
    --------
    接收 JSON: { "bird_name": "麻雀", "question": "它吃什么？" }
    返回 JSON: { "answer": "..." }
    
    回答策略（优先级从高到低）：
        1. 关键词匹配知识库
        2. 调用DeepSeek大模型API动态生成
        3. 返回默认兜底回答
    """
    # 1. 解析请求
    data = request.get_json(silent=True)
    if not data:
        return jsonify({"answer": "请求格式错误，请传入JSON数据"}), 400

    bird_name = data.get("bird_name", "").strip()
    question = data.get("question", "").strip()

    if not bird_name:
        return jsonify({"answer": "请提供鸟类名称"}), 400
    if not question:
        return jsonify({"answer": "请提出您的问题"}), 400

    logger.info(f"收到问答请求 - 鸟类: {bird_name}, 问题: {question}")

    # 2. 策略一：关键词匹配知识库
    answer = match_knowledge(bird_name, question)
    if answer:
        logger.info(f"知识库匹配成功: {answer[:40]}...")
        return jsonify({"answer": answer})

    # 3. 策略二：调用DeepSeek大模型
    answer = ask_deepseek(bird_name, question)
    if answer:
        return jsonify({"answer": answer})

    # 4. 策略三：兜底回答
    fallback = f"抱歉，我暂时无法回答关于{bird_name}的这个问题：「{question}」。请换个问法或稍后再试。"
    logger.warning(f"所有回答策略均失败，返回兜底回答")
    return jsonify({"answer": fallback})


@app.route("/api/history", methods=["GET"])
def api_history():
    """
    查询识别历史记录
    返回最近10条记录
    """
    records = get_history(limit=10)
    return jsonify({"success": True, "records": records})


@app.route("/api/health", methods=["GET"])
def api_health():
    """健康检查接口"""
    return jsonify({
        "status": "ok",
        "local_model_loaded": is_model_loaded(),

        "deepseek_configured": bool(DEEPSEEK_API_KEY),
        "timestamp": datetime.now().isoformat(),
    })


# ======================================================================
#  主入口
# ======================================================================

if __name__ == "__main__":
    # 初始化数据库
    init_db()

    # 尝试加载本地模型（不阻塞启动）
    logger.info("🔄 尝试加载本地模型...")
    model_loaded = load_local_model()
    if model_loaded:
        logger.info("✅ 本地模型加载成功，将使用本地模型进行识别")
    else:
        logger.warning("=" * 60)
        logger.warning("⚠️  本地模型未加载（models/best_model.pth 不存在）")
        logger.warning("  请在 training/ 目录下运行 train.py 训练模型")

        logger.warning("=" * 60)












    if not DEEPSEEK_API_KEY:
        logger.warning("=" * 60)
        logger.warning("⚠️  DeepSeek API Key 未配置！")
        logger.warning("  问答功能的知识库匹配仍可正常工作")
        logger.warning("  申请地址: https://platform.deepseek.com/")
        logger.warning("=" * 60)

    # 尝试加载 YOLO 模型
    logger.info("🔄 尝试加载 YOLO 型号（鸟类轮廓检测）...")
    yolo_loaded = load_yolo_model()
    if yolo_loaded:
        logger.info("✅ YOLO 模型加载成功，将进行鸟类轮廓检测")
    else:
        logger.warning("⚠️  YOLO 模型未加载，将跳过目标检测（不影响分类）")

    # 启动Flask服务
    logger.info("🚀 启动鸟类识别与问答系统后端...")
    logger.info("📡 服务地址: http://localhost:5000")
    logger.info("📋 API文档:")


    logger.info("  POST /api/recognize  - 鸟类识别（YOLO轮廓检测 + ResNet50分类）")
    logger.info("  POST /api/ask       - 智能问答")
    logger.info("  GET  /api/history   - 历史记录")
    logger.info("  GET  /api/health    - 健康检查")

    app.run(host="0.0.0.0", port=5000, debug=True)
