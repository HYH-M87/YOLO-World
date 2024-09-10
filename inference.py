import PIL.Image
import cv2
import supervision as sv
import os

import numpy as np
import torch
from mmengine.config import Config
from mmengine.runner import Runner
from mmengine.dataset import Compose
from mmengine.runner.amp import autocast
from torchvision.ops import nms


cfg = Config.fromfile(
        "configs/pretrain/yolo_world_v2_l_vlpan_bn_2e-3_100e_4x8gpus_obj365v1_goldg_train_1280ft_lvis_minival.py"
    )
cfg.work_dir = "."
cfg.load_from = "pretrained_weights/yolo_world_v2_l_obj365v1_goldg_pretrain_1280ft-9babe3f6.pth"
runner = Runner.from_cfg(cfg)
runner = Runner.from_cfg(cfg)
runner.call_hook("before_run")
runner.load_or_resume()
pipeline = cfg.test_dataloader.dataset.pipeline
runner.pipeline = Compose(pipeline)

# run model evaluation
runner.model.eval()


def colorstr(*input):
    """
        Helper function for style logging
    """
    *args, string = input if len(input) > 1 else ("bold", input[0])
    colors = {"bold": "\033[1m"}

    return "".join(colors[x] for x in args) + f"{string}"

bounding_box_annotator = sv.BoxAnnotator()
label_annotator = sv.LabelAnnotator(text_position=sv.Position.CENTER)
mask_annotator = sv.MaskAnnotator()

# class_names = ("person, bicycle, car, motorcycle, airplane, bus, train, truck, boat, "
#                "traffic light, fire hydrant, stop sign, parking meter, bench, bird, "
#                "cat, dog, horse, sheep, cow, elephant, bear, zebra, giraffe, "
#                "backpack, umbrella, handbag, tie, suitcase, frisbee, skis, snowboard, "
#                "sports ball, kite, baseball bat, baseball glove, skateboard, "
#                "surfboard, tennis racket, bottle, wine glass, cup, fork, knife, "
#                "spoon, bowl, banana, apple, sandwich, orange, broccoli, carrot, "
#                "hot dog, pizza, donut, cake, chair, couch, potted plant, bed, "
#                "dining table, toilet, tv, laptop, mouse, remote, keyboard, "
#                "cell phone, microwave, oven, toaster, sink, refrigerator, book, "
#                "clock, vase, scissors, teddy bear, hair drier, toothbrush")

# class_names2 = ("dog, eye, tongue, ear, leash")
class_names = ("car, pedestrian, bus, truck, tank, armored car")
class_inplace = ["compact car", "pedestrian", "bus", "lorry", "tank", "armored car"]

def run_image(
        runner,
        input_image,
        output_path,
        max_num_boxes=100,
        score_thr=0.05,
        nms_thr=0.5,
):
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    # 输出图像名字
    output_image = os.path.join(output_path, input_image.split('/')[-1])

    texts = [[t.strip()] for t in class_names.split(",")] + [[" "]]
    data_info = runner.pipeline(dict(img_id=0, img_path=input_image,
                                     texts=texts))

    data_batch = dict(
        inputs=data_info["inputs"].unsqueeze(0),
        data_samples=[data_info["data_samples"]],
    )

    with autocast(enabled=False), torch.no_grad():
        output = runner.model.test_step(data_batch)[0]
        runner.model.class_names = texts
        pred_instances = output.pred_instances

    # nms
    pred_instances = pred_instances[pred_instances.scores.float() > score_thr]
    keep_idxs = nms(pred_instances.bboxes, pred_instances.scores, iou_threshold=nms_thr)
    pred_instances = pred_instances[keep_idxs]
    

    if len(pred_instances.scores) > max_num_boxes:
        indices = pred_instances.scores.float().topk(max_num_boxes)[1]
        pred_instances = pred_instances[indices]
    output.pred_instances = pred_instances

    # predictions
    pred_instances = pred_instances.cpu().numpy()

    if 'masks' in pred_instances:
        masks = pred_instances['masks']
    else:
        masks = None
        
    detections = sv.Detections(
        xyxy=pred_instances['bboxes'],
        class_id=pred_instances['labels'],
        confidence=pred_instances['scores']
    )

    # label ids with confidence scores
    labels = [
        f"{class_inplace[class_id]} {confidence:0.2f}"
        for class_id, confidence
        in zip(detections.class_id, detections.confidence)
    ]

    # draw bounding box with label
    image = PIL.Image.open(input_image)
    svimage = np.array(image)
    svimage = bounding_box_annotator.annotate(svimage, detections)
    svimage = label_annotator.annotate(svimage, detections, labels)
    if masks is not None:
        svimage = mask_annotator.annotate(image, detections)

    # save output image
    cv2.imwrite(output_image, svimage[:, :, ::-1])
    print(f"Results saved to {colorstr('bold', output_image)}")

    return svimage[:, :, ::-1]

if __name__ == '__main__':
    input_path = '../images240520'
    output_path = '../predict_1500'
    image = [os.path.join(input_path, i) for i in os.listdir(input_path)]
    for item in image:
        run_image(runner, item, output_path)
    print("successful!!!")