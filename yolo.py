import colorsys
import os
import time

import numpy as np
import torch
import torch.nn as nn
from PIL import ImageDraw, ImageFont, Image

from nets.yolo import YoloBody
from utils.utils import (cvtColor, get_anchors, get_classes, preprocess_input,
                         resize_image, show_config)
from utils.utils_bbox import DecodeBox

'''
训练自己的数据集必看注释！
'''


class YOLO(object):
    _defaults = {
        # --------------------------------------------------------------------------#
        #   使用自己训练好的模型进行预测一定要修改model_path和classes_path！
        #   model_path指向logs文件夹下的权值文件，classes_path指向model_data下的txt
        #
        #   训练好后logs文件夹下存在多个权值文件，选择验证集损失较低的即可。
        #   验证集损失较低不代表mAP较高，仅代表该权值在验证集上泛化性能较好。
        #   如果出现shape不匹配，同时要注意训练时的model_path和classes_path参数的修改
        # --------------------------------------------------------------------------#
        "model_path": 'model_data/yolov7_tiny_weights.pth',
        "classes_path": 'model_data/myclass.txt',
        # ---------------------------------------------------------------------#
        #   anchors_path代表先验框对应的txt文件，一般不修改。
        #   anchors_mask用于帮助代码找到对应的先验框，一般不修改。
        # ---------------------------------------------------------------------#
        "anchors_path": 'model_data/my_anchors.txt',
        "anchors_mask": [[6, 7, 8], [3, 4, 5], [0, 1, 2]],
        # ---------------------------------------------------------------------#
        #   输入图片的大小，必须为32的倍数。
        # ---------------------------------------------------------------------#
        "input_shape": [640, 640],
        # ---------------------------------------------------------------------#
        #   只有得分大于置信度的预测框会被保留下来
        # ---------------------------------------------------------------------#
        "confidence": 0.5,
        # ---------------------------------------------------------------------#
        #   非极大抑制所用到的nms_iou大小
        # ---------------------------------------------------------------------#
        "nms_iou": 0.3,
        # ---------------------------------------------------------------------#
        #   该变量用于控制是否使用letterbox_image对输入图像进行不失真的resize，
        #   在多次测试后，发现关闭letterbox_image直接resize的效果更好
        # ---------------------------------------------------------------------#
        "letterbox_image": True,
        # -------------------------------#
        #   是否使用Cuda
        #   没有GPU可以设置成False
        # -------------------------------#
        "cuda": True,
    }

    @classmethod
    def get_defaults(cls, n):
        if n in cls._defaults:
            return cls._defaults[n]
        else:
            return "Unrecognized attribute name '" + n + "'"

    # ---------------------------------------------------#
    #   初始化YOLO
    # ---------------------------------------------------#
    def __init__(self, **kwargs):
        self.__dict__.update(self._defaults)
        for name, value in kwargs.items():
            setattr(self, name, value)
            self._defaults[name] = value

            # ---------------------------------------------------#
        #   获得种类和先验框的数量
        # ---------------------------------------------------#
        self.class_names, self.num_classes = get_classes(self.classes_path)
        self.anchors, self.num_anchors = get_anchors(self.anchors_path)
        self.bbox_util = DecodeBox(self.anchors, self.num_classes, (self.input_shape[0], self.input_shape[1]),
                                   self.anchors_mask)

        # ---------------------------------------------------#
        #   画框设置不同的颜色
        # ---------------------------------------------------#
        hsv_tuples = [(x / self.num_classes, 1., 1.) for x in range(self.num_classes)]
        self.colors = list(map(lambda x: colorsys.hsv_to_rgb(*x), hsv_tuples))
        self.colors = list(map(lambda x: (int(x[0] * 255), int(x[1] * 255), int(x[2] * 255)), self.colors))
        self.generate()

        show_config(**self._defaults)

    # ---------------------------------------------------#
    #   生成模型
    # ---------------------------------------------------#
    def generate(self, onnx=False):
        # ---------------------------------------------------#
        #   建立yolo模型，载入yolo模型的权重
        # ---------------------------------------------------#
        self.net = YoloBody(self.anchors_mask, self.num_classes)
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.net.load_state_dict(torch.load(self.model_path, map_location=device))
        self.net = self.net.fuse().eval()
        print('{} model, and classes loaded.'.format(self.model_path))
        if not onnx:
            if self.cuda:
                self.net = nn.DataParallel(self.net)
                self.net = self.net.cuda()

    # ---------------------------------------------------#
    #   检测图片
    # ---------------------------------------------------#
    def detect_image(self, image, crop=False, count=False):
        # ---------------------------------------------------#
        #   计算输入图片的高和宽
        # ---------------------------------------------------#
        image_shape = np.array(np.shape(image)[0:2])
        # ---------------------------------------------------------#
        #   在这里将图像转换成RGB图像，防止灰度图在预测时报错。
        #   代码仅仅支持RGB图像的预测，所有其它类型的图像都会转化成RGB
        # ---------------------------------------------------------#
        image = cvtColor(image)
        # ---------------------------------------------------------#
        #   给图像增加灰条，实现不失真的resize
        #   也可以直接resize进行识别
        # ---------------------------------------------------------#
        image_data = resize_image(image, (self.input_shape[1], self.input_shape[0]), self.letterbox_image)
        # ---------------------------------------------------------#
        #   添加上batch_size维度
        # ---------------------------------------------------------#
        image_data = np.expand_dims(np.transpose(preprocess_input(np.array(image_data, dtype='float32')), (2, 0, 1)), 0)

        with torch.no_grad():
            images = torch.from_numpy(image_data)
            if self.cuda:
                images = images.cuda()
            # ---------------------------------------------------------#
            #   将图像输入网络当中进行预测！
            # ---------------------------------------------------------#
            outputs = self.net(images)
            outputs = self.bbox_util.decode_box(outputs)
            # ---------------------------------------------------------#
            #   将预测框进行堆叠，然后进行非极大抑制
            # ---------------------------------------------------------#
            results = self.bbox_util.non_max_suppression(torch.cat(outputs, 1), self.num_classes, self.input_shape,
                                                         image_shape, self.letterbox_image, conf_thres=self.confidence,
                                                         nms_thres=self.nms_iou)

            if results[0] is None:
                return image

            top_label = np.array(results[0][:, 6], dtype='int32')
            top_conf = results[0][:, 4] * results[0][:, 5]
            top_boxes = results[0][:, :4]
        # ---------------------------------------------------------#
        #   设置字体与边框厚度
        # ---------------------------------------------------------#
        font = ImageFont.truetype(font='model_data/simhei.ttf',
                                  size=np.floor(3e-2 * image.size[1] + 0.5).astype('int32'))
        thickness = int(max((image.size[0] + image.size[1]) // np.mean(self.input_shape), 1))
        # ---------------------------------------------------------#
        #   计数
        # ---------------------------------------------------------#
        if count:
            print("top_label:", top_label)
            classes_nums = np.zeros([self.num_classes])
            for i in range(self.num_classes):
                num = np.sum(top_label == i)
                if num > 0:
                    print(self.class_names[i], " : ", num)
                classes_nums[i] = num
            print("classes_nums:", classes_nums)
        # ---------------------------------------------------------#
        #   是否进行目标的裁剪
        # ---------------------------------------------------------#
        if crop:
            for i, c in list(enumerate(top_boxes)):
                top, left, bottom, right = top_boxes[i]
                top = max(0, np.floor(top).astype('int32'))
                left = max(0, np.floor(left).astype('int32'))
                bottom = min(image.size[1], np.floor(bottom).astype('int32'))
                right = min(image.size[0], np.floor(right).astype('int32'))

                dir_save_path = "img_crop"
                if not os.path.exists(dir_save_path):
                    os.makedirs(dir_save_path)
                crop_image = image.crop([left, top, right, bottom])
                crop_image.save(os.path.join(dir_save_path, "crop_" + str(i) + ".png"), quality=95, subsampling=0)
                print("save crop_" + str(i) + ".png to " + dir_save_path)
        # ---------------------------------------------------------#
        #   图像绘制
        # ---------------------------------------------------------#
        for i, c in list(enumerate(top_label)):
            predicted_class = self.class_names[int(c)]
            box = top_boxes[i]
            score = top_conf[i]

            top, left, bottom, right = box

            top = max(0, np.floor(top).astype('int32'))
            left = max(0, np.floor(left).astype('int32'))
            bottom = min(image.size[1], np.floor(bottom).astype('int32'))
            right = min(image.size[0], np.floor(right).astype('int32'))

            label = '{} {:.2f}'.format(predicted_class, score)
            draw = ImageDraw.Draw(image)
            label_size = draw.textsize(label, font)
            label = label.encode('utf-8')
            print(label, top, left, bottom, right)

            if top - label_size[1] >= 0:
                text_origin = np.array([left, top - label_size[1]])
            else:
                text_origin = np.array([left, top + 1])

            for i in range(thickness):
                draw.rectangle([left + i, top + i, right - i, bottom - i], outline=self.colors[c])
            draw.rectangle([tuple(text_origin), tuple(text_origin + label_size)], fill=self.colors[c])
            draw.text(text_origin, str(label, 'UTF-8'), fill=(0, 0, 0), font=font)
            del draw

        return image

    def get_FPS(self, image, test_interval):
        image_shape = np.array(np.shape(image)[0:2])
        # ---------------------------------------------------------#
        #   在这里将图像转换成RGB图像，防止灰度图在预测时报错。
        #   代码仅仅支持RGB图像的预测，所有其它类型的图像都会转化成RGB
        # ---------------------------------------------------------#
        image = cvtColor(image)
        # ---------------------------------------------------------#
        #   给图像增加灰条，实现不失真的resize
        #   也可以直接resize进行识别
        # ---------------------------------------------------------#
        image_data = resize_image(image, (self.input_shape[1], self.input_shape[0]), self.letterbox_image)
        # ---------------------------------------------------------#
        #   添加上batch_size维度
        # ---------------------------------------------------------#
        image_data = np.expand_dims(np.transpose(preprocess_input(np.array(image_data, dtype='float32')), (2, 0, 1)), 0)

        with torch.no_grad():
            images = torch.from_numpy(image_data)
            if self.cuda:
                images = images.cuda()
            # ---------------------------------------------------------#
            #   将图像输入网络当中进行预测！
            # ---------------------------------------------------------#
            outputs = self.net(images)
            outputs = self.bbox_util.decode_box(outputs)
            # ---------------------------------------------------------#
            #   将预测框进行堆叠，然后进行非极大抑制
            # ---------------------------------------------------------#
            results = self.bbox_util.non_max_suppression(torch.cat(outputs, 1), self.num_classes, self.input_shape,
                                                         image_shape, self.letterbox_image, conf_thres=self.confidence,
                                                         nms_thres=self.nms_iou)

        t1 = time.time()
        for _ in range(test_interval):
            with torch.no_grad():
                # ---------------------------------------------------------#
                #   将图像输入网络当中进行预测！
                # ---------------------------------------------------------#
                outputs = self.net(images)
                outputs = self.bbox_util.decode_box(outputs)
                # ---------------------------------------------------------#
                #   将预测框进行堆叠，然后进行非极大抑制
                # ---------------------------------------------------------#
                results = self.bbox_util.non_max_suppression(torch.cat(outputs, 1), self.num_classes, self.input_shape,
                                                             image_shape, self.letterbox_image,
                                                             conf_thres=self.confidence, nms_thres=self.nms_iou)

        t2 = time.time()
        tact_time = (t2 - t1) / test_interval
        return tact_time

    def detect_heatmap(self, image, heatmap_save_path):
        import cv2
        import matplotlib.pyplot as plt
        def sigmoid(x):
            y = 1.0 / (1.0 + np.exp(-x))
            return y

        # ---------------------------------------------------------#
        #   在这里将图像转换成RGB图像，防止灰度图在预测时报错。
        #   代码仅仅支持RGB图像的预测，所有其它类型的图像都会转化成RGB
        # ---------------------------------------------------------#
        image = cvtColor(image)
        # ---------------------------------------------------------#
        #   给图像增加灰条，实现不失真的resize
        #   也可以直接resize进行识别
        # ---------------------------------------------------------#
        image_data = resize_image(image, (self.input_shape[1], self.input_shape[0]), self.letterbox_image)
        # ---------------------------------------------------------#
        #   添加上batch_size维度
        # ---------------------------------------------------------#
        image_data = np.expand_dims(np.transpose(preprocess_input(np.array(image_data, dtype='float32')), (2, 0, 1)), 0)

        with torch.no_grad():
            images = torch.from_numpy(image_data)
            if self.cuda:
                images = images.cuda()
            # ---------------------------------------------------------#
            #   将图像输入网络当中进行预测！
            # ---------------------------------------------------------#
            outputs = self.net(images)

        plt.imshow(image, alpha=1)
        plt.axis('off')
        mask = np.zeros((image.size[1], image.size[0]))
        for sub_output in outputs:
            sub_output = sub_output.cpu().numpy()
            b, c, h, w = np.shape(sub_output)
            sub_output = np.transpose(np.reshape(sub_output, [b, 3, -1, h, w]), [0, 3, 4, 1, 2])[0]
            score = np.max(sigmoid(sub_output[..., 4]), -1)
            score = cv2.resize(score, (image.size[0], image.size[1]))
            normed_score = (score * 255).astype('uint8')
            mask = np.maximum(mask, normed_score)

        plt.imshow(mask, alpha=0.5, interpolation='nearest', cmap="jet")

        plt.axis('off')
        plt.subplots_adjust(top=1, bottom=0, right=1, left=0, hspace=0, wspace=0)
        plt.margins(0, 0)
        plt.savefig(heatmap_save_path, dpi=200, bbox_inches='tight', pad_inches=-0.1)
        print("Save to the " + heatmap_save_path)
        plt.show()

    def convert_to_onnx(self, simplify, model_path):
        import onnx
        self.generate(onnx=True)

        im = torch.zeros(1, 3, *self.input_shape).to('cpu')  # image size(1, 3, 512, 512) BCHW
        input_layer_names = ["images"]
        output_layer_names = ["output"]

        # Export the model
        print(f'Starting export with onnx {onnx.__version__}.')
        torch.onnx.export(self.net,
                          im,
                          f=model_path,
                          verbose=False,
                          opset_version=12,
                          training=torch.onnx.TrainingMode.EVAL,
                          do_constant_folding=True,
                          input_names=input_layer_names,
                          output_names=output_layer_names,
                          dynamic_axes=None)

        # Checks
        model_onnx = onnx.load(model_path)  # load onnx model
        onnx.checker.check_model(model_onnx)  # check onnx model

        # Simplify onnx
        if simplify:
            import onnxsim
            print(f'Simplifying with onnx-simplifier {onnxsim.__version__}.')
            model_onnx, check = onnxsim.simplify(
                model_onnx,
                dynamic_input_shape=False,
                input_shapes=None)
            assert check, 'assert check failed'
            onnx.save(model_onnx, model_path)

        print('Onnx model save as {}'.format(model_path))

    def get_map_txt(self, image_id, image, class_names, map_out_path):
        f = open(os.path.join(map_out_path, "detection-results/" + image_id + ".txt"), "w", encoding='utf-8')
        image_shape = np.array(np.shape(image)[0:2])
        # ---------------------------------------------------------#
        #   在这里将图像转换成RGB图像，防止灰度图在预测时报错。
        #   代码仅仅支持RGB图像的预测，所有其它类型的图像都会转化成RGB
        # ---------------------------------------------------------#
        image = cvtColor(image)
        # ---------------------------------------------------------#
        #   给图像增加灰条，实现不失真的resize
        #   也可以直接resize进行识别
        # ---------------------------------------------------------#
        image_data = resize_image(image, (self.input_shape[1], self.input_shape[0]), self.letterbox_image)
        # ---------------------------------------------------------#
        #   添加上batch_size维度
        # ---------------------------------------------------------#
        image_data = np.expand_dims(np.transpose(preprocess_input(np.array(image_data, dtype='float32')), (2, 0, 1)), 0)

        with torch.no_grad():
            images = torch.from_numpy(image_data)
            if self.cuda:
                images = images.cuda()
            # ---------------------------------------------------------#
            #   将图像输入网络当中进行预测！
            # ---------------------------------------------------------#
            outputs = self.net(images)
            outputs = self.bbox_util.decode_box(outputs)
            # ---------------------------------------------------------#
            #   将预测框进行堆叠，然后进行非极大抑制
            # ---------------------------------------------------------#
            results = self.bbox_util.non_max_suppression(torch.cat(outputs, 1), self.num_classes, self.input_shape,
                                                         image_shape, self.letterbox_image, conf_thres=self.confidence,
                                                         nms_thres=self.nms_iou)

            if results[0] is None:
                return

            top_label = np.array(results[0][:, 6], dtype='int32')
            top_conf = results[0][:, 4] * results[0][:, 5]
            top_boxes = results[0][:, :4]

        for i, c in list(enumerate(top_label)):
            predicted_class = self.class_names[int(c)]
            box = top_boxes[i]
            score = str(top_conf[i])

            top, left, bottom, right = box
            if predicted_class not in class_names:
                continue

            f.write("%s %s %s %s %s %s\n" % (
                predicted_class, score[:6], str(int(left)), str(int(top)), str(int(right)), str(int(bottom))))

        f.close()
        return

    def get_results(self, image_id):
        image_path = os.path.join('data/BMPImage', '%s.bmp' % image_id)
        # 预处理
        image = Image.open(image_path)
        image_shape = np.array(np.shape(image)[0:2])
        image = cvtColor(image)
        image_data = resize_image(image, (self.input_shape[1], self.input_shape[0]), self.letterbox_image)
        image_data = np.expand_dims(
            np.transpose(preprocess_input(np.array(image_data, dtype='float32')), (2, 0, 1)), 0)

        with torch.no_grad():
            images = torch.from_numpy(image_data)
            if self.cuda:
                images = images.cuda()
            outputs = self.net(images)
            outputs = self.bbox_util.decode_box(outputs)
            # ---------------------------------------------------------#
            #   将预测框进行堆叠，然后进行非极大抑制
            # ---------------------------------------------------------#
            results = self.bbox_util.non_max_suppression(torch.cat(outputs, 1), self.num_classes, self.input_shape,
                                                         image_shape, self.letterbox_image, conf_thres=self.confidence,
                                                         nms_thres=self.nms_iou)
        if results[0] is None:
            return

        label = results[0][:, 6]  # class index
        conf = results[0][:, 4] * results[0][:, 5]  # score
        boxes = results[0][:, [1, 0, 3, 2]]  # y1,x1,y2,x2

        return torch.from_numpy(label), torch.from_numpy(conf), torch.from_numpy(boxes)

    def get_fusemap_txt(self, images_id, classes_name, results_pt):
        label_0, conf_0, boxes_0 = self.get_results(images_id[0])
        count_0 = torch.ones(len(boxes_0))  # 上一个角度计数情况，这里初始化为第1个视角的计数
        label_9, conf_9, boxes_9 = self.get_results(images_id[8])
        count_9 = torch.ones(len(boxes_9))  # 第9个视角的计数情况
        for i in range(1, 5):
            label_next, conf_next, boxes_next = self.get_results(images_id[i])
            pair_iou = bboxes_iou(boxes_0, boxes_next)  # 计算前一个角度和后一个角度预测框的iou情况
            if pair_iou.shape[1] == 0:  # 如果下一个角度没有预测框,直接跳过到下一个角度
                continue
            pair_iou_mask = ~ (torch.sum(pair_iou > 0, dim=1).bool())
            # 0     0.5    0        0     0.5   0       0       0       0
            # 0.5   0      0.5  ->  0.5   0     0 ->    0.5     0       0
            # 0     0.4    0        0     0.4   0       0       0.4     0
            # 横向找最大，纵向找最多。
            ious, indices = torch.max(pair_iou, dim=1)  # indices 当前框加到下一个角度框的索引
            pair_iou_copy = torch.zeros((len(boxes_0), len(boxes_next)))
            pair_iou_copy[range(len(boxes_0)), indices] = ious  # 此时已经保证每一行至多一个数不为0
            for i in range(len(boxes_next)):
                non_zero_idx = torch.nonzero(pair_iou_copy[:, i])
                if len(non_zero_idx) < 2:
                    continue
                _, indice = torch.max(count_0[non_zero_idx], dim=0)  # 判断哪个框的数量较多
                max_indice = non_zero_idx[indice]  # 得到数量最大的前一个框
                pair_iou_copy[:, i] = 0
                pair_iou_copy[max_indice, i] = 1
            ious, indices = torch.max(pair_iou_copy, dim=1)
            mask = ious.bool()  # 如果最大值Iou是0,那么该框不添加到下一个框,那么问题来了。前面的框不算了嘛

            boxes_0 = torch.cat((boxes_next, boxes_0[pair_iou_mask]), dim=0)
            conf_0 = torch.cat((conf_next, conf_0[pair_iou_mask]), dim=0)
            label_0 = torch.cat((label_next, label_0[pair_iou_mask]), dim=0)

            count_next = torch.ones(len(boxes_next))
            count_next = torch.cat((count_next, count_0[pair_iou_mask]))  # 将没有重叠框的计数 cat到下一个角度。
            count_next[indices[mask]] += count_0[mask]
            count_0 = count_next

        for j in range(7, 4, -1):
            label_last, conf_last, boxes_last = self.get_results(images_id[j])
            pair_iou = bboxes_iou(boxes_9, boxes_last)  # 计算前一个角度和后一个角度预测框的iou情况
            pair_iou_mask = ~ (torch.sum(pair_iou > 0, dim=1).bool())
            if pair_iou.shape[1] == 0:  # 说明下一个角度没有框
                continue

            ious, indices = torch.max(pair_iou, dim=1)  # 170的时候pair_iou的Shape为(1,0) 所以无法在dim=1上进行Iou的计算
            pair_iou_copy = torch.zeros((len(boxes_9), len(boxes_last)))
            pair_iou_copy[range(len(boxes_9)), indices] = ious  # 此时已经保证每一行至多一个数不为0
            for i in range(len(boxes_last)):
                non_zero_idx = torch.nonzero(pair_iou_copy[:, i])
                if len(non_zero_idx) < 2:
                    continue
                _, indice = torch.max(count_9[non_zero_idx], dim=0)  # 判断哪个框的数量较多
                max_indice = non_zero_idx[indice]  # 得到数量最大的前一个框
                pair_iou_copy[:, i] = 0
                pair_iou_copy[max_indice, i] = 1
            ious, indices = torch.max(pair_iou_copy, dim=1)
            mask = ious.bool()

            boxes_9 = torch.cat((boxes_last, boxes_9[pair_iou_mask]), dim=0)
            conf_9 = torch.cat((conf_last, conf_9[pair_iou_mask]), dim=0)
            label_9 = torch.cat((label_last, label_9[pair_iou_mask]), dim=0)

            count_last = torch.ones(len(boxes_last))
            count_last = torch.cat((count_last, count_9[pair_iou_mask]))
            count_last[indices[mask]] += count_9[mask]
            count_9 = count_last

        pair_iou = bboxes_iou(boxes_9, boxes_0)
        pair_iou_mask = ~ (torch.sum(pair_iou > 0, dim=1).bool())

        ious, indices = torch.max(pair_iou, dim=1)
        pair_iou_copy = torch.zeros((len(boxes_9), len(boxes_0)))
        pair_iou_copy[range(len(boxes_9)), indices] = ious  # 此时已经保证每一行至多一个数不为0
        for i in range(len(boxes_0)):
            non_zero_idx = torch.nonzero(pair_iou_copy[:, i])
            if len(non_zero_idx) < 2:
                continue
            _, indice = torch.max(count_9[non_zero_idx], dim=0)  # 判断哪个框的数量较多
            max_indice = non_zero_idx[indice]  # 得到数量最大的前一个框
            pair_iou_copy[:, i] = 0
            pair_iou_copy[max_indice, i] = 1
        ious, indices = torch.max(pair_iou_copy, dim=1)
        mask = ious.bool()

        boxes = torch.cat((boxes_0, boxes_9[pair_iou_mask]), dim=0).int()
        labels = torch.cat((label_0, label_9[pair_iou_mask]), dim=0)
        conf = torch.cat((conf_0, conf_9[pair_iou_mask]), dim=0)

        count = torch.cat((count_0, count_9[pair_iou_mask]))
        count[indices[mask]] += count_9[mask]
        # 进行过滤预测框
        # count_mask = (count > 1) & (count > 2) | (conf > 0.7)
        count_mask = count > 1
        boxes = boxes[count_mask]
        labels = labels[count_mask]
        conf = conf[count_mask]
        # print(count)

        with open(os.path.join(results_pt, "detection-results/" + images_id[4] + ".txt"), "w") as new_f:
            for m, box in enumerate(boxes):
                # boxes : ymin,xmin,ymax,xmax
                new_f.write(
                    "%s %s %s %s %s %s\n" % (
                        classes_name[int(labels[m])], str(float("%.4f" % conf[m])), str(int(box[0])), str(int(box[1])),
                        str(int(box[2])),
                        str(int(box[3]))))


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
