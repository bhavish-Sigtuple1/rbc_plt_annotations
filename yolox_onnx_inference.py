import onnxruntime
import numpy as np
import cv2



class YOLOX_ONNX:
    def __init__(self, onnx_model_file):
        self.session = onnxruntime.InferenceSession(onnx_model_file)
        self.input_name = self.session.get_inputs()[0].name
        self.ratio = 1

    def preprocess_input(self, img, input_size=(416, 416), swap=(0, 3, 1, 2)):
        # if len(img.shape) == 3:
        #     padded_img = np.ones((input_size[0], input_size[1], 3), dtype=np.uint8) * 114
        # else:
        #     padded_img = np.ones(input_size, dtype=np.uint8) * 114

        # r = min(input_size[0] / img.shape[0], input_size[1] / img.shape[1])
        # resized_img = cv2.resize(
        #     img,
        #     (int(img.shape[1] * r), int(img.shape[0] * r)),
        #     interpolation=cv2.INTER_LINEAR,
        # ).astype(np.uint8)
        # padded_img[: int(img.shape[0] * r), : int(img.shape[1] * r)] = resized_img


        padded_img = img.transpose(swap)
        padded_img = np.ascontiguousarray(padded_img, dtype=np.float32)
        self.ratio = 1.0
        return padded_img, 1.0

    def nms(self, boxes, scores, nms_thr):
        x1 = boxes[:, 0]
        y1 = boxes[:, 1]
        x2 = boxes[:, 2]
        y2 = boxes[:, 3]

        areas = (x2 - x1 + 1) * (y2 - y1 + 1)
        order = scores.argsort()[::-1]

        keep = []
        while order.size > 0:
            i = order[0]
            keep.append(i)
            xx1 = np.maximum(x1[i], x1[order[1:]])
            yy1 = np.maximum(y1[i], y1[order[1:]])
            xx2 = np.minimum(x2[i], x2[order[1:]])
            yy2 = np.minimum(y2[i], y2[order[1:]])

            w = np.maximum(0.0, xx2 - xx1 + 1)
            h = np.maximum(0.0, yy2 - yy1 + 1)
            inter = w * h
            ovr = inter / (areas[i] + areas[order[1:]] - inter)

            inds = np.where(ovr <= nms_thr)[0]
            order = order[inds + 1]

        return keep
    
    def multiclass_nms_class_agnostic(self, boxes, scores, nms_thr, score_thr):
        cls_inds = scores.argmax(1)
        cls_scores = scores[np.arange(len(cls_inds)), cls_inds]

        valid_score_mask = cls_scores > score_thr
        if valid_score_mask.sum() == 0:
            return None
        valid_scores = cls_scores[valid_score_mask]
        valid_boxes = boxes[valid_score_mask]
        valid_cls_inds = cls_inds[valid_score_mask]
        keep = self.nms(valid_boxes, valid_scores, nms_thr)
        if keep:
            dets = np.concatenate(
                [valid_boxes[keep], valid_scores[keep, None], valid_cls_inds[keep, None]], 1
            )
        return dets
    
    def multiclass_nms_class_aware(self, boxes, scores, nms_thr, score_thr):
        final_dets = []
        num_classes = scores.shape[1]
        for cls_ind in range(num_classes):
            cls_scores = scores[:, cls_ind]
            valid_score_mask = cls_scores > score_thr
            if valid_score_mask.sum() == 0:
                continue
            else:
                valid_scores = cls_scores[valid_score_mask]
                valid_boxes = boxes[valid_score_mask]
                keep = self.nms(valid_boxes, valid_scores, nms_thr)
                if len(keep) > 0:
                    cls_inds = np.ones((len(keep), 1)) * cls_ind
                    dets = np.concatenate(
                        [valid_boxes[keep], valid_scores[keep, None], cls_inds], 1
                    )
                    final_dets.append(dets)
        if len(final_dets) == 0:
            return None
        return np.concatenate(final_dets, 0)


    def multiclass_nms(self, boxes, scores, nms_thr, score_thr, class_agnostic=True):
        if class_agnostic:
            nms_method = self.multiclass_nms_class_agnostic
        else:
            nms_method = self.multiclass_nms_class_aware
        return nms_method(boxes, scores, nms_thr, score_thr)
    
    def postprocess_output(self, outputs_list, conf_threshold=0.1, iou_threshold=0.45, p6=False, img_size=(416, 416)):
        dets_list = []
        for outputs in outputs_list:
            outputs = np.expand_dims(outputs, axis=0)
            grids = []
            expanded_strides = []

            if not p6:
                strides = [8, 16, 32]
            else:
                strides = [8, 16, 32, 64]

            hsizes = [img_size[0] // stride for stride in strides]
            wsizes = [img_size[1] // stride for stride in strides]

            for hsize, wsize, stride in zip(hsizes, wsizes, strides):
                xv, yv = np.meshgrid(np.arange(wsize), np.arange(hsize))
                grid = np.stack((xv, yv), 2).reshape(1, -1, 2)
                grids.append(grid)
                shape = grid.shape[:2]
                expanded_strides.append(np.full((*shape, 1), stride))

            grids = np.concatenate(grids, 1)
            expanded_strides = np.concatenate(expanded_strides, 1)
            outputs[..., :2] = (outputs[..., :2] + grids) * expanded_strides
            outputs[..., 2:4] = np.exp(outputs[..., 2:4]) * expanded_strides
            predictions = outputs[0]
            boxes = predictions[:, :4]
            scores = predictions[:, 4:5] * predictions[:, 5:]
            boxes_xyxy = np.ones_like(boxes)
            boxes_xyxy[:, 0] = boxes[:, 0] - boxes[:, 2] / 2.
            boxes_xyxy[:, 1] = boxes[:, 1] - boxes[:, 3] / 2.
            boxes_xyxy[:, 2] = boxes[:, 0] + boxes[:, 2] / 2.
            boxes_xyxy[:, 3] = boxes[:, 1] + boxes[:, 3] / 2.
            boxes_xyxy /= self.ratio
            dets = self.multiclass_nms(boxes_xyxy, scores, nms_thr=iou_threshold, score_thr=conf_threshold)
            dets_list.append(dets)
        return dets_list

    def infer(self, image):
        processed_image = self.preprocess_input(image[:, :, :, ::-1])
        outputs = self.session.run(None, {self.input_name: processed_image[0]})[0]
        detections = self.postprocess_output(outputs)
        return detections
