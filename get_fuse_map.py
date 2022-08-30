import os
import numpy as np
import xml.etree.ElementTree as ET
from utils.utils import get_classes
from utils.utils_map import get_map
import torch
from tqdm import tqdm
from yolo import YOLO

# 将9张图片的真实框融合到正视图上,用一个最小覆盖框进行覆盖
# 从结果来看,图像是左右转动，因此ymin和ymax几乎不变，我们fuse的时候只需要取最小的ymin,和最大ymax即可
# 因为图像是左右转动，因此xmin和xmax的在不同的角度变化较大。

# 真实框融合，对于正视图有的框，那就直接用正视图的框进行表示，如果正视图上没有，那就进行融合，反正就是要将所有的框先融合到这张图上。
# 怎么进行融合? 从左右两边层层循序渐进的遍历。如果第一个视角和第二角度图像进行比较,如果第一个视角的框与第二个视角的框重叠了，那么就舍弃，如果没有就加入到第二个视角的boxes。
# 如此中间视角的框就像冒泡一样来到中间视角，分别从第一个视角和第九个视角出发,向中间冒泡

# 测试集图片的名称
test_image_pt = "data/Main/test.txt"
# xml文件存放路径
xml_pt = "data/Annotations"
# 数据集 种类名称 和 种类数量
classes_name, classes_num = get_classes("model_data/myclass.txt")
# 输出结果存放路径
results_pt = "fusion_map_out"
# 模型预测后处理confidence和nms阈值
confidence = 0.25
nms_iou = 0.1
# ap的iou阈值
MINOVERLAP = 0.5

# 模式：
#   mode = 0, 包含下面三个全过程。
#   mode = 1, 获取真实框的融合结果
#   mode = 2, 获取预测框的融合结果
#   mode = 3, 得到融合后的性能指标
mode = 0


def get_annotation(image_id):
    in_file = open(os.path.join(xml_pt, "%s.xml" % image_id), encoding='utf-8')
    tree = ET.parse(in_file)
    root = tree.getroot()
    boxes = []
    ids = []
    for obj in root.iter('object'):  # 遍历xml文件中 object的标签
        difficult = 0
        if obj.find('difficult') is not None:
            difficult = obj.find('difficult').text
        cls = obj.find('name').text  # 获取obj的name
        if cls not in classes_name or int(difficult) == 1:
            continue
        ids.append(classes_name.index(cls))
        xmlbox = obj.find('bndbox')  # 得到目标的坐标
        b = (int(float(xmlbox.find('xmin').text)), int(float(xmlbox.find('ymin').text)),
             int(float(xmlbox.find('xmax').text)), int(float(xmlbox.find('ymax').text)))
        boxes.append(b)
    return torch.tensor(ids), torch.tensor(boxes)


def bboxes_iou(bboxes_a, bboxes_b):  # 计算pair_iou
    bboxes_a = torch.tensor(bboxes_a)
    bboxes_b = torch.tensor(bboxes_b)
    tl = torch.max(bboxes_a[:, None, :2], bboxes_b[:, :2])  # [num_a,num_b,2]返回两两之间xmin和ymin较大的值
    br = torch.min(bboxes_a[:, None, 2:], bboxes_b[:, 2:])  # [num_a,num_b,2]返回两两之间xmax和ymax较大的值
    area_a = torch.prod(bboxes_a[:, 2:] - bboxes_a[:, :2], 1)  # 得到bboxes_a 所有框的面积
    area_b = torch.prod(bboxes_b[:, 2:] - bboxes_b[:, :2], 1)  # 得到bboxes_b 所有框的面积
    en = (tl < br).type(tl.type()).prod(dim=2)  # [num_a,num_b] 判断两个gt框是否相交,如果没有相交,要将相交面积置0，不然可能会出现负数，或者负数和负数相乘的情况
    area_i = torch.prod(br - tl, 2) * en  # [num_a,num_b]两两之间的相交区域
    return area_i / (area_a[:, None] + area_b - area_i)  # 得到两两之间的Iou


if __name__ == '__main__':
    if not os.path.exists(results_pt):
        os.mkdir(results_pt)
    if not os.path.exists(os.path.join(results_pt, "ground-truth")):
        os.mkdir(os.path.join(results_pt, "ground-truth"))
    if not os.path.exists(os.path.join(results_pt, "detection-results")):
        os.mkdir(os.path.join(results_pt, "detection-results"))

    images_id_list = open(test_image_pt, encoding="utf-8").read().strip().split()
    images_id_array = np.array(images_id_list).reshape(-1, 9)  # (200,9)

    if mode == 0 or mode == 1:
        print("fusion ground truth box")
        for images_id in tqdm(images_id_array):  # 取组
            ids_0, boxes_0 = get_annotation(images_id[0])  # 第1个角度的框
            ids_9, boxes_9 = get_annotation(images_id[8])  # 第9个角度的框
            for j in range(4):  # 取图片
                ids_next, boxes_next = get_annotation(images_id[j + 1])  # 下一个角度图像
                pair_iou = bboxes_iou(boxes_0, boxes_next)  # 计算前后角度图像gt框的pair_iou
                pair_iou_mask = ~ (torch.sum(pair_iou > 0, dim=1).bool())  # 取出当前角度有的框，而下一个角度没有的框
                boxes_0 = torch.cat((boxes_next, boxes_0[pair_iou_mask]), dim=0)
                ids_0 = torch.cat((ids_next, ids_0[pair_iou_mask]), dim=0)

            for k in range(7, 4, -1):
                ids_last, boxes_last = get_annotation(images_id[k])
                pair_iou = bboxes_iou(boxes_9, boxes_last)
                pair_iou_mask = ~ (torch.sum(pair_iou > 0, dim=1).bool())
                boxes_9 = torch.cat((boxes_last, boxes_9[pair_iou_mask]), dim=0)
                ids_9 = torch.cat((ids_last, ids_9[pair_iou_mask]), dim=0)

            pair_iou = bboxes_iou(boxes_9, boxes_0)
            pair_iou_mask = ~ (torch.sum(pair_iou > 0, dim=1).bool())
            boxes = torch.cat((boxes_0, boxes_9[pair_iou_mask]), dim=0)
            ids = torch.cat((ids_0, ids_9[pair_iou_mask]), dim=0)
            with open(os.path.join(results_pt, "ground-truth/" + images_id[4] + ".txt"), "w") as new_f:
                for m, box in enumerate(boxes):
                    new_f.write(
                        "%s %s %s %s %s\n" % (classes_name[int(ids[m])], str(int(box[0])), str(int(box[1])),
                                              str(int(box[2])), str(int(box[3]))))
        print("Get ground truth result done")
    # 先获取单帧图像的预测框,然后进行融合.加入了过滤算法
    if mode == 0 or mode == 2:
        print("Load model")
        yolo = YOLO(confidence=confidence, nms_iou=nms_iou)
        print("Load model done")
        for i, images_id in enumerate(images_id_array):
            yolo.get_fusemap_txt(images_id, classes_name, results_pt=results_pt)
            print("--------{}-------".format(i + 1))

        print("Get predict result done.")

    # # 获取预测结果
    if mode == 0 or mode == 3:
        print("Get map.")
        get_map(MINOVERLAP, True, score_threhold=confidence, path=results_pt, nms=nms_iou)
        print("Get map done.")

    # 对框进行排序
    # for i, images_id in (enumerate(images_id_array)):
    #     for image_id in images_id:
    #         _, _, boxes = yolo.get_results(image_id)
    #         boxes_list = boxes.tolist()
    #         boxes_list = [list(map(int, x)) for x in boxes_list]
    #         for j in range(len(boxes_list) - 1):
    #             for k in range(len(boxes_list) - 1 - j):
    #                 if boxes_list[k][1] > boxes_list[k + 1][1]:
    #                     temp = boxes_list[k]
    #                     boxes_list[k] = boxes_list[k + 1]
    #                     boxes_list[k + 1] = temp
    #         print(boxes_list)
    #
    #     print("----------{}--------".format(i + 1))
    # if mode == 4:
    #     boxes_a = [[145, 387, 191, 454]]
    #
    #     boxes_b = [[140, 387, 187, 453]]
    #     pair_iou = bboxes_iou(boxes_a, boxes_b)
    #     if pair_iou.shape[0] == 1 or pair_iou.shape[1] == 1:
    #         print("hello world")
