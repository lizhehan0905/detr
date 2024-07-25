# 训练相关基本操作
## 环境
+ python
+ torch
+ 其他
```
pip install -r requirements.txt -i  https://pypi.tuna.tsinghua.edu.cn/simple

```
+ 注意panopticapi和pycocotools可能会安装报错
## 数据集
+ 数据格式为coco格式
```
├── annotations
│    ├── instances_train2017.json
│    └── instances_val2017.json
│ 
├── train2017
│   
└── val2017
   

```
## 命令行执行main.py

+ 需要去models/detr.py里面的build函数（317行）手动将num_classes写死成你的类别数+1
+ 预训练权重主要是resnet50的骨干网络和resnet101的骨干网络
### 训练相关
+ --lr 学习率
+ --lr_backbone 骨干网络学习率
+ --batch-size 视显存大小而定
+ --epochs 训练代数
+ --backbone 默认是resnet50
### 数据集相关
+ --coco_path 数据集地址
+ --output_dir 输出地址
+ --resume 需要指定断点重训的checkpoint地址，迁移学习也是在这里实现的
+ --num_workers 指定0肯定可以
### 评估
+ --eval
### 推理
+ 官方没有推理代码,需要修改好--backbone、--coco_path、--outputdir、--resume以及main函数中的image_file_path
+ 服务器上运行注意注释掉cv2.imshow类的显示代码
```
import argparse
import random
import time
from pathlib import Path
import numpy as np
import torch
from models import build_model
from PIL import Image
import os
import torchvision
from torchvision.ops.boxes import batched_nms
import cv2

#-------------------------------------------------------------------------设置参数
def get_args_parser():
    parser = argparse.ArgumentParser('Set transformer detector', add_help=False)
    parser.add_argument('--lr', default=1e-4, type=float)
    parser.add_argument('--lr_backbone', default=1e-5, type=float)
    parser.add_argument('--batch_size', default=2, type=int)
    parser.add_argument('--weight_decay', default=1e-4, type=float)
    parser.add_argument('--epochs', default=300, type=int)
    parser.add_argument('--lr_drop', default=200, type=int)
    parser.add_argument('--clip_max_norm', default=0.1, type=float,
                        help='gradient clipping max norm')

    # Model parameters
    parser.add_argument('--frozen_weights', type=str, default=None,
                        help="Path to the pretrained model. If set, only the mask head will be trained")
    # * Backbone
    parser.add_argument('--backbone', default='resnet101', type=str,
                        help="Name of the convolutional backbone to use")
    parser.add_argument('--dilation', action='store_true',
                        help="If true, we replace stride with dilation in the last convolutional block (DC5)")
    parser.add_argument('--position_embedding', default='sine', type=str, choices=('sine', 'learned'),
                        help="Type of positional embedding to use on top of the image features")

    # * Transformer
    parser.add_argument('--enc_layers', default=6, type=int,
                        help="Number of encoding layers in the transformer")
    parser.add_argument('--dec_layers', default=6, type=int,
                        help="Number of decoding layers in the transformer")
    parser.add_argument('--dim_feedforward', default=2048, type=int,
                        help="Intermediate size of the feedforward layers in the transformer blocks")
    parser.add_argument('--hidden_dim', default=256, type=int,
                        help="Size of the embeddings (dimension of the transformer)")
    parser.add_argument('--dropout', default=0.1, type=float,
                        help="Dropout applied in the transformer")
    parser.add_argument('--nheads', default=8, type=int,
                        help="Number of attention heads inside the transformer's attentions")
    parser.add_argument('--num_queries', default=100, type=int,
                        help="Number of query slots")
    parser.add_argument('--pre_norm', action='store_true')

    # * Segmentation
    parser.add_argument('--masks', action='store_true',
                        help="Train segmentation head if the flag is provided")

    # Loss
    parser.add_argument('--no_aux_loss', dest='aux_loss', default='False',
                        help="Disables auxiliary decoding losses (loss at each layer)")
    # * Matcher
    parser.add_argument('--set_cost_class', default=1, type=float,
                        help="Class coefficient in the matching cost")
    parser.add_argument('--set_cost_bbox', default=5, type=float,
                        help="L1 box coefficient in the matching cost")
    parser.add_argument('--set_cost_giou', default=2, type=float,
                        help="giou box coefficient in the matching cost")
    # * Loss coefficients
    parser.add_argument('--mask_loss_coef', default=1, type=float)
    parser.add_argument('--dice_loss_coef', default=1, type=float)
    parser.add_argument('--bbox_loss_coef', default=5, type=float)
    parser.add_argument('--giou_loss_coef', default=2, type=float)
    parser.add_argument('--eos_coef', default=0.1, type=float,
                        help="Relative classification weight of the no-object class")

    # dataset parameters
    parser.add_argument('--dataset_file', default='coco')
    parser.add_argument('--coco_path', type=str,default="coco")
    parser.add_argument('--coco_panoptic_path', type=str)
    parser.add_argument('--remove_difficult', action='store_true')

    parser.add_argument('--output_dir', default='inference_demo/inference_output',
                        help='path where to save, empty for no saving')
    parser.add_argument('--device', default='cuda',
                        help='device to use for training / testing')
    parser.add_argument('--seed', default=42, type=int)
    parser.add_argument('--resume', default='inference_demo/weights/detr-r101-2c7b67e5.pth', help='resume from checkpoint')
    parser.add_argument('--start_epoch', default=0, type=int, metavar='N',
                        help='start epoch')
    parser.add_argument('--eval', default="True")
    parser.add_argument('--num_workers', default=2, type=int)

    # distributed training parameters
    parser.add_argument('--world_size', default=1, type=int,
                        help='number of distributed processes')
    parser.add_argument('--dist_url', default='env://', help='url used to set up distributed training')
    return parser


def box_cxcywh_to_xyxy(x):
    #将DETR的检测框坐标(x_center,y_cengter,w,h)转化成coco数据集的检测框坐标(x0,y0,x1,y1)
    x_c, y_c, w, h = x.unbind(1)
    b = [(x_c - 0.5 * w), (y_c - 0.5 * h),
         (x_c + 0.5 * w), (y_c + 0.5 * h)]
    return torch.stack(b, dim=1)

def rescale_bboxes(out_bbox, size):
    #把比例坐标乘以图像的宽和高，变成真实坐标
    img_w, img_h = size
    b = box_cxcywh_to_xyxy(out_bbox)
    b = b * torch.tensor([img_w, img_h, img_w, img_h], dtype=torch.float32)
    return b

def filter_boxes(scores, boxes, confidence=0.7, apply_nms=True, iou=0.5):
    #筛选出真正的置信度高的框
    keep = scores.max(-1).values > confidence
    scores, boxes = scores[keep], boxes[keep]

    if apply_nms:
        top_scores, labels = scores.max(-1)
        keep = batched_nms(boxes, top_scores, labels, iou)
        scores, boxes = scores[keep], boxes[keep]

    return scores, boxes

# COCO classes
CLASSES = [
    'N/A', 'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus',
    'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'N/A',
    'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse',
    'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'N/A', 'backpack',
    'umbrella', 'N/A', 'N/A', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis',
    'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove',
    'skateboard', 'surfboard', 'tennis racket', 'bottle', 'N/A', 'wine glass',
    'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich',
    'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake',
    'chair', 'couch', 'potted plant', 'bed', 'N/A', 'dining table', 'N/A',
    'N/A', 'toilet', 'N/A', 'tv', 'laptop', 'mouse', 'remote', 'keyboard',
    'cell phone', 'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'N/A',
    'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier',
    'toothbrush'
]

def plot_one_box(x, img, color=None, label=None, line_thickness=1):
    #把检测框画到图片上
    tl = line_thickness or round(0.002 * (img.shape[0] + img.shape[1]) / 2) + 1  # line/font thickness
    color = color or [random.randint(0, 255) for _ in range(3)]
    c1, c2 = (int(x[0]), int(x[1])), (int(x[2]), int(x[3]))
    cv2.rectangle(img, c1, c2, color, thickness=tl, lineType=cv2.LINE_AA)
    if label:
        tf = max(tl - 1, 1)  # font thickness
        t_size = cv2.getTextSize(label, 0, fontScale=tl / 3, thickness=tf)[0]
        c2 = c1[0] + t_size[0], c1[1] - t_size[1] - 3
        cv2.rectangle(img, c1, c2, color, -1, cv2.LINE_AA)  # filled
        cv2.putText(img, label, (c1[0], c1[1] - 2), 0, tl / 3, [225, 255, 255], thickness=tf, lineType=cv2.LINE_AA)


def main(args):
    print(args)
    device = torch.device(args.device)

    #------------------------------------导入网络
    #下面的criterion是算损失函数要用的，推理用不到,postprocessors是解码用的，这里也没有用，用的是自己的。
    model, criterion, postprocessors = build_model(args) 

    #------------------------------------加载权重
    checkpoint = torch.load(args.resume, map_location='cuda')
    model.load_state_dict(checkpoint['model'])

    #------------------------------------把权重加载到gpu或cpu上
    model.to(device)

    #------------------------------------打印出网络的参数大小
    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print("parameters:",n_parameters)
    
    #------------------------------------设置好存储输出结果的文件夹
    output_dir = Path(args.output_dir)

    #-----------------------------------读取数据集,进行推理
    image_Totensor=torchvision.transforms.ToTensor()
    image_file_path = os.listdir("inference_demo/detect_demo")
    image_set = []

    for image_item in image_file_path:
        print("inference_image:",image_item)
        image_path = os.path.join("inference_demo/detect_demo",image_item)
        image = Image.open(image_path)
        image_tensor = image_Totensor(image)
        image_tensor = torch.reshape(image_tensor,[-1,image_tensor.shape[0],image_tensor.shape[1],image_tensor.shape[2]])
        image_tensor=image_tensor.to(device)
        time1 = time.time()
        inference_result = model(image_tensor)
        time2 = time.time()
        print("inference_time:",time2-time1)
        probas = inference_result['pred_logits'].softmax(-1)[0, :, :-1].cpu()
        bboxes_scaled = rescale_bboxes(inference_result['pred_boxes'][0, ].cpu(),(image_tensor.shape[3],image_tensor.shape[2]))
        scores, boxes = filter_boxes(probas,bboxes_scaled)
        scores = scores.data.numpy()
        boxes = boxes.data.numpy()
        for i in range(boxes.shape[0]):
            class_id = scores[i].argmax()
            label = CLASSES[class_id]
            confidence = scores[i].max()
            text = f"{label} {confidence:.3f}"
            image = np.array(image)
            plot_one_box(boxes[i],image,label=text)
        cv2.imshow("images",image)
        cv2.waitKey(1)
        image=Image.fromarray(image)
        image.save(os.path.join(args.output_dir,image_item))

if __name__ == '__main__':
    parser = argparse.ArgumentParser('DETR training and evaluation script', parents=[get_args_parser()])
    args = parser.parse_args()
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    main(args)

```
## 绘图
+ 在util工具下的Plot_utils.py中调用执行绘图
```
if __name__ == '__main__':
    files = list(Path('../output/eval').glob('*.pth'))
    plot_precision_recall(files)
    plt.show()
    plot_logs(logs=Path('../outputs/log/'),fields=('class_error', 'loss_bbox_unscaled', 'mAP'), ewm_col=0, log_name='log.txt')
    plt.show()
```

