import numpy as np


def change_wh(box_xy):
    box_wh = np.zeros_like(box_xy)

    box_wh[..., 0] = (box_xy[..., 2] + box_xy[..., 0]) * 0.5
    box_wh[..., 1] = (box_xy[..., 3] + box_xy[..., 1]) * 0.5
    box_wh[..., 2] = box_xy[..., 2] - box_xy[..., 0]
    box_wh[..., 3] = box_xy[..., 3] - box_xy[..., 1]

    return box_wh

def change_xy(box_wh):
    box_xy = np.zeros_like(box_wh)

    box_xy[..., 0] = box_wh[..., 0] - (box_wh[..., 2]) * 0.5
    box_xy[..., 1] = box_wh[..., 1] - (box_wh[..., 3]) * 0.5
    box_xy[..., 2] = box_wh[..., 0] + (box_wh[..., 2]) * 0.5
    box_xy[..., 3] = box_wh[..., 1] + (box_wh[..., 3]) * 0.5

    return box_xy

def IOU(box_a, box_b):
    xa = np.maximum(box_a[0], box_b[..., 0])
    ya = np.maximum(box_a[1], box_b[..., 1])
    xb = np.minimum(box_a[2], box_b[..., 2])
    yb = np.minimum(box_a[3], box_b[..., 3])

    inter = np.maximum(0, xb - xa) * np.maximum(0, yb - ya)
    a_area = (box_a[2] - box_a[0]) * (box_a[3] - box_a[1])
    b_area = (box_b[..., 2] - box_b[..., 0]) * (box_b[..., 3] - box_b[..., 1])

    return inter / (a_area + b_area - inter + 1e-5)

def IOU2(box_a, box_b):
    m = box_a.shape[0]
    n = box_b.shape[0]

    box_a = np.tile(np.expand_dims(box_a, axis= 1), (1, n, 1))
    box_b = np.tile(np.expand_dims(box_b, axis= 0), (m, 1, 1))

    xa = np.maximum(box_a[..., 0], box_b[..., 0])
    ya = np.maximum(box_a[..., 1], box_b[..., 1])
    xb = np.minimum(box_a[..., 2], box_b[..., 2])
    yb = np.minimum(box_a[..., 3], box_b[..., 3])

    inter = np.maximum(0, xb - xa) * np.maximum(0, yb - ya)
    a_area = (box_a[..., 2] - box_a[..., 0]) * (box_a[..., 3] - box_a[..., 1])
    b_area = (box_b[..., 2] - box_b[..., 0]) * (box_b[..., 3] - box_b[..., 1])

    return inter / (a_area + b_area - inter + 1e-8)

def nms(scores, bboxes, thresh, max_boxes):
    sort_idx = np.argsort(scores)[::-1]
    keep = np.zeros(len(sort_idx), np.bool)

    i = 0
    while len(sort_idx) > 1:
        keep[sort_idx[0]] = True
        i += 1

        if i == max_boxes:
            break

        ovps = IOU(bboxes[sort_idx[0]], bboxes[sort_idx[1:]])

        pick = np.where(ovps < thresh)[0] + 1
        sort_idx = sort_idx[pick]

    return keep

def resize_bbox(bboxes, origin_size, new_size):
    w_ratio = float(origin_size[1]) / new_size[1]
    h_ratio = float(origin_size[0]) / new_size[0]

    re_bbox = np.zeros_like(bboxes, np.int32)
    re_bbox[..., 0] = np.round(bboxes[..., 0] * w_ratio).astype(np.int32)
    re_bbox[..., 1] = np.round(bboxes[..., 1] * h_ratio).astype(np.int32)
    re_bbox[..., 2] = np.round(bboxes[..., 2] * w_ratio).astype(np.int32)
    re_bbox[..., 3] = np.round(bboxes[..., 3] * h_ratio).astype(np.int32)

    return re_bbox


class Anchor:
    def __init__(self, C):
        self.C = C
        self.baseAnchors = []

        scales = np.linspace(C.scale_range[0], C.scale_range[1], C.n_scale)

        for k, scale in enumerate(scales[:-1]):
            current_anchor = []
            for ratio in C.ratios[k]:
                if ratio == 1:
                    w = h = C.imageSize * scale
                    current_anchor.append([w, h])
                    w = h = C.imageSize * np.sqrt(scale * scales[k + 1])
                    current_anchor.append([w, h])
                else:
                    w = scale * C.imageSize * np.sqrt(ratio)
                    h = scale * C.imageSize / np.sqrt(ratio)
                    current_anchor.append([w, h])
            self.baseAnchors.append(np.array(current_anchor))

    def GetAnchors(self, f_size):
        anchors_wh = []
        anchors_xy = []
        for i, baseAnchor in enumerate(self.baseAnchors):
            n_anchors = baseAnchor.shape[0]
            height, width = f_size[i][1:3]
            anchor_wh = np.zeros((height, width, n_anchors, 4), np.float32)

            wstep = self.C.imageSize / width
            hstep = self.C.imageSize / height
            cx = np.linspace(wstep * 0.5, (0.5 + width - 1) * wstep, width)
            cy = np.linspace(hstep * 0.5, (0.5 + height - 1) * hstep, height)
            cx_grid, cy_grid = np.meshgrid(cx, cy)
            cx_grid = np.expand_dims(cx_grid, -1)
            cy_grid = np.expand_dims(cy_grid, -1)

            anchor_wh[:, :, :, 0] = np.tile(cx_grid, (1, 1, n_anchors))
            anchor_wh[:, :, :, 1] = np.tile(cy_grid, (1, 1, n_anchors))
            anchor_wh[:, :, :, 2] = baseAnchor[:, 0]
            anchor_wh[:, :, :, 3] = baseAnchor[:, 1]

            ##########################################
            #           Normalization coord          #
            ##########################################
            #anchor_wh /= self.C.imageSize

            anchor_xy = change_xy(anchor_wh)

            anchors_wh.append(anchor_wh.reshape((-1, 4)))
            anchors_xy.append(anchor_xy.reshape((-1, 4)))

        return np.concatenate(anchors_wh, axis=0), np.concatenate(anchors_xy, axis=0)


DEBUG = 1

class y_encoder:
    def __init__(self, anchors, C):
        self.anchors_xy = anchors[1]
        self.anchors_wh = anchors[0]
        self.n_classes = len(C.classes) + 1
        self.class_vector = np.eye(self.n_classes)
        self.max_thresh = C.max_thresh
        self.variances = C.variances
        self.img_size = C.imageSize

    if DEBUG == 0:
        def __call__(self, labels):
            assignment = np.zeros((self.n_batches, self.n_anchors, self.n_classes + 4 + 4), np.float32)
            assignment[..., :-8] = self.class_vector[-1]
            tmp_iou = np.zeros(self.n_anchors, np.float32)

            for i in range(self.n_batches):
                tmp_iou.fill(0)
                n_labels = self.class_vector[labels[i][:, 0]]
                bbox_xy = labels[i][:, 1:]
                bbox_wh = change_wh(bbox_xy)
                for j in range(labels[i].shape[0]):
                    iou = IOU(bbox_xy[j], self.anchors_xy)
                    max_mask = np.logical_and(iou > self.max_thresh, iou > tmp_iou)

                    if not max_mask.any():
                        max_mask[iou.argmax()] = True

                    assignment[i, max_mask, -8:-6] = (bbox_wh[j, :2] - self.anchors_wh[max_mask, :2]) / self.anchors_wh[max_mask, 2:]
                    assignment[i, max_mask, -6:-4] = np.log(bbox_wh[j, 2:] / self.anchors_wh[max_mask, 2:])
                    #assignment[i, max_mask, -8:-4] /= self.variances
                    assignment[i, max_mask, :-8] = n_labels[j]

                    tmp_iou[:] = np.maximum(iou, tmp_iou)
            return assignment

    elif DEBUG == 1:
        def encode_box(self, box):
            ##########################################
            #           Normalization coord          #
            ##########################################
            box_wh = change_wh(box) #/ self.img_size

            iou = IOU(box, self.anchors_xy)
            encoded_box = np.zeros((len(self.anchors_wh), 4 + 1))

            best_idx = np.where(iou > self.max_thresh)[0]
            if len(best_idx) == 0:
                best_idx = np.argmax(iou)

            encoded_box[best_idx, -1] = iou[best_idx]

            encoded_box[best_idx, 0] = (box_wh[0] - self.anchors_wh[best_idx, 0]) / self.anchors_wh[best_idx, 2]
            encoded_box[best_idx, 1] = (box_wh[1] - self.anchors_wh[best_idx, 1]) / self.anchors_wh[best_idx, 3]
            encoded_box[best_idx, 2] = np.log(box_wh[2] / self.anchors_wh[best_idx, 2])
            encoded_box[best_idx, 3] = np.log(box_wh[3] / self.anchors_wh[best_idx, 3])

            encoded_box[best_idx, :-1] /= self.variances

            return encoded_box

        '''
        [batch, boxes, 1 + classes + 4 + 4]
        '''
        def __call__(self, labels):
            nAnchor = len(self.anchors_wh)
            assignment = np.zeros((nAnchor, self.n_classes + 4))

            assignment[:, :-4] = self.class_vector[-1]

            encoded_boxes = np.apply_along_axis(self.encode_box, -1, labels[..., :-1])

            indices = np.arange(nAnchor)
            best_iou_rows = encoded_boxes[:, :, -1].argmax(axis=0)
            best_iou_cols = np.nonzero(encoded_boxes[best_iou_rows, indices, -1] > 0)[0]
            best_iou_rows = best_iou_rows[best_iou_cols]

            nLabels = labels[..., -1].reshape([1, -1])
            assignment[best_iou_cols, -4:] = encoded_boxes[best_iou_rows, best_iou_cols, :4]
            assignment[best_iou_cols, :-4] = self.class_vector[nLabels[0, best_iou_rows]]

            return assignment

    elif DEBUG == 2:
        '''
        [batch, boxes, classes + 4 + 4]
        '''
        def __call__(self, labels):
            assignment = np.zeros((self.n_batches, self.n_anchors, self.n_classes + 4 + 4), np.float32)
            assignment[:, :, :-8] = self.class_vector[-1]
            for i in range(self.n_batches):
                bbox_xy = labels[i][:, 1:]
                bbox_wh = change_wh(bbox_xy)
                n_label = labels[i][:, 0]

                onehot_labels = self.class_vector[n_label]

                # [n_bbox, n_anchors]
                iou = IOU2(bbox_xy, self.anchors_xy)

                best_idx = self.choice_best_anchor(iou)
                assignment[i, best_idx, :-8] = onehot_labels
                assignment[i, best_idx, -8:-6] = (bbox_wh[:, :2] - self.anchors_wh[best_idx, :2]) / self.anchors_wh[best_idx, 2:]
                assignment[i, best_idx, -6:-4] = np.log(bbox_wh[:, 2:] / self.anchors_wh[best_idx, 2:])
                #assignment[i, best_idx, -8:-4] /= self.variances

                gt_idx, anchor_idx = self.choice_multi_anchor(iou)
                assignment[i, anchor_idx, :-8] = onehot_labels[gt_idx]
                assignment[i, anchor_idx, -8:-6] = (bbox_wh[gt_idx, :2] - self.anchors_wh[anchor_idx, :2]) / self.anchors_wh[anchor_idx, 2:]
                assignment[i, anchor_idx, -6:-4] = np.log(bbox_wh[gt_idx, 2:] / self.anchors_wh[anchor_idx, 2:])
                #assignment[i, anchor_idx, -8:-4] /= self.variances

            return assignment

        def choice_best_anchor(self, ious):
            ious = np.copy(ious)
            n_bbox = ious.shape[0]
            row_indices = list(range(n_bbox))

            best_anchor = np.zeros(n_bbox, np.int32)
            for _ in range(n_bbox):
                col_idxes = np.argmax(ious, axis=-1)
                col_values = ious[row_indices, col_idxes]
                best_row_idx = np.argmax(col_values)

                anchor_idx = col_idxes[best_row_idx]
                best_anchor[best_row_idx] = anchor_idx

                ious[:, anchor_idx] = 0
                ious[best_row_idx] = 0

            return best_anchor

        def choice_multi_anchor(self, ious):
            col_indices = list(range(self.n_anchors))

            row_idxes = np.argmax(ious, axis=0)
            row_values = ious[row_idxes, col_indices]
            col_valid_idx = np.nonzero(row_values > self.max_thresh)[0]

            return row_idxes[col_valid_idx], col_valid_idx

    elif DEBUG == 3:
        def __call__(self, labels):
            conf_assign = np.zeros((self.n_batches, self.n_anchors, 1 + self.n_classes), np.float32)
            local_assign = np.zeros((self.n_batches, self.n_anchors, 1 + 4), np.float32)

            conf_assign[..., 0] = 1
            tmp_iou = np.zeros(self.n_anchors, np.float32)

            for i in range(self.n_batches):
                tmp_iou.fill(0)
                for j in range(labels[i].shape[0]):
                    bbox_xy = labels[i][j, 1:]
                    bbox_wh = change_wh(bbox_xy)
                    label = labels[i][j, 0]

                    iou = IOU(bbox_xy, self.anchors_xy)
                    max_mask = np.logical_and(iou > self.max_thresh, iou > tmp_iou)

                    if not max_mask.any():
                        max_mask[iou.argmax()] = True

                    conf_assign[i, max_mask, 0] = 0
                    conf_assign[i, max_mask, 1:] = self.class_vector[label]

                    local_assign[i, max_mask, 4] = 1
                    local_assign[i, max_mask, :2] = (bbox_wh[:2] - self.anchors_wh[max_mask, :2]) / self.anchors_wh[max_mask, 2:]
                    local_assign[i, max_mask, 2:4] = np.log(bbox_wh[2:] / self.anchors_wh[max_mask, 2:])
                    #local_assign[i, max_mask, :4] /= self.variances

                    tmp_iou[:] = np.maximum(iou, tmp_iou)
            return conf_assign, local_assign

    elif DEBUG == 4:
        '''
        [batch, boxes, classes + 4 + 4]
        '''
        def __call__(self, labels):
            assignment = np.zeros((self.n_batches, self.n_anchors, self.n_classes + 4 + 4), np.float32)
            assignment[:, :, :-8] = self.class_vector[-1]
            for i in range(self.n_batches):
                bbox_xy = labels[i][:, 1:] #/ self.img_size
                bbox_wh = change_wh(bbox_xy)
                n_label = labels[i][:, 0]

                onehot_labels = self.class_vector[n_label]

                # [n_bbox, n_anchors]
                iou = IOU2(bbox_xy, self.anchors_xy)

                gt_idx, anchor_idx = self.choice_multi_anchor(iou)
                assignment[i, anchor_idx, :-8] = onehot_labels[gt_idx]
                assignment[i, anchor_idx, -8:-6] = (bbox_wh[gt_idx, :2] - self.anchors_wh[anchor_idx, :2]) / self.anchors_wh[anchor_idx, 2:]
                assignment[i, anchor_idx, -6:-4] = np.log(bbox_wh[gt_idx, 2:] / self.anchors_wh[anchor_idx, 2:])
                #assignment[i, anchor_idx, -8:-4] /= self.variances

            return assignment

        def choice_multi_anchor(self, ious):
            col_indices = list(range(self.n_anchors))

            row_idxes = np.argmax(ious, axis=0)
            row_values = ious[row_idxes, col_indices]
            col_valid_idx = np.nonzero(row_values > self.max_thresh)[0]

            return row_idxes[col_valid_idx], col_valid_idx


























