from collections import Counter
from scipy.optimize import linear_sum_assignment
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import ImageGrid
from PIL import ImageDraw

import torch
from torch import nn
import torch.nn.functional as F
import torchvision
from torchvision.ops.boxes import box_area
from torchvision.transforms import ToPILImage
from torchmetrics.detection import MeanAveragePrecision

from d2l import torch as d2l


def box_iou(boxes1, boxes2):
    """
    Compute the Intersection over Union (IoU) between two sets of bounding boxes. The format of the bounding boxes
    should be in (x1, y1, x2, y2) format.

    Args:
        boxes1 (torch.Tensor): A tensor of shape (N, 4) in (x1, y1, x2, y2) format.
        boxes2 (torch.Tensor): A tensor of shape (M, 4) in (x1, y1, x2, y2) format.

    Returns:
        tuple[torch.Tensor, torch.Tensor]: A tuple of (iou, union) where:
            iou (torch.Tensor): A tensor of shape (N, M) containing the pairwise IoU values
            between the boxes in boxes1 and boxes2.
            union (torch.Tensor): A tensor of shape (N, M) containing the pairwise union
            areas between the boxes in boxes1 and boxes2.
    """
    # Calculate boxes area
    area1 = box_area(boxes1)  # [N,]
    area2 = box_area(boxes2)  # [M,]

    # Compute the coordinates of the intersection of each pair of bounding boxes
    lt = torch.max(boxes1[:, None, :2], boxes2[:, :2])  # [N,M,2]
    rb = torch.min(boxes1[:, None, 2:], boxes2[:, 2:])  # [N,M,2]
    # Need clamp(min=0) in case they do not intersect, then we want intersection to be 0
    wh = (rb - lt).clamp(min=0)  # [N,M,2]
    inter = wh[:, :, 0] * wh[:, :, 1]  # [N,M]

    # Since the size of the variables is different, pytorch broadcast them
    # area1[:, None] converts size from [N,] to [N,1] to help broadcasting
    union = area1[:, None] + area2 - inter  # [N,M]

    iou = inter / union
    return iou, union


def generalized_box_iou(boxes1, boxes2):
    """
    Computes the generalized box intersection over union (IoU) between two sets of bounding boxes.
    The IoU is defined as the area of overlap between the two bounding boxes divided by the area of union.

    Args:
        boxes1: A tensor containing the coordinates of the bounding boxes for the first set.
            Shape: [batch_size, num_boxes, 4]
        boxes2: A tensor containing the coordinates of the bounding boxes for the second set.
            Shape: [batch_size, num_boxes, 4]

    Returns:
        A tensor containing the generalized IoU between `boxes1` and `boxes2`.
            Shape: [batch_size, num_boxes1, num_boxes2]
    """
    # Check for degenerate boxes that give Inf/NaN results
    assert (boxes1[:, 2:] >= boxes1[:, :2]).all()
    assert (boxes2[:, 2:] >= boxes2[:, :2]).all()

    # Calculate the IoU and union of each pair of bounding boxes
    # TODO: put your code here (~1 line)
    iou, union = box_iou(boxes1, boxes2)

    # Compute the coordinates of the intersection of each pair of bounding boxes
    lt = torch.min(boxes1[:, None, :2], boxes2[:, :2])
    rb = torch.max(boxes1[:, None, 2:], boxes2[:, 2:])
    wh = (rb - lt).clamp(min=0)  # [N,M,2]

    # Compute the area of the bounding box that encloses both input boxes
    C = wh[:, :, 0] * wh[:, :, 1]

    # TODO: put your code here (~1 line)
    return iou - (C - union) / C


class HungarianMatcher(nn.Module):
    """This class computes an assignment between the targets and the predictions of the network

    For efficiency reasons, the targets don't include the no_object. Because of this, in general,
    there are more predictions than targets. In this case, we do a 1-to-1 matching of the best predictions,
    while the others are un-matched (and thus treated as non-objects).
    """

    def __init__(self, cost_class: float = 1, cost_bbox: float = 1, cost_giou: float = 1):
        """Creates the matcher

        Params:
            cost_class: This is the relative weight of the classification error in the matching cost
            cost_bbox: This is the relative weight of the L1 error of the bounding box coordinates in the matching cost
            cost_giou: This is the relative weight of the giou loss of the bounding box in the matching cost
        """
        super().__init__()
        self.cost_class = cost_class
        self.cost_bbox = cost_bbox
        self.cost_giou = cost_giou
        assert cost_class != 0 or cost_bbox != 0 or cost_giou != 0, "all costs cant be 0"

    @torch.no_grad()
    def forward(self, outputs, targets):
        """Performs the matching

        Params:
            outputs: This is a dict that contains at least these entries:
                 "pred_logits": Tensor of dim [batch_size, num_queries, num_classes] with the classification logits
                 "pred_boxes": Tensor of dim [batch_size, num_queries, 4] with the predicted box coordinates

            targets: This is a list of targets (len(targets) = batch_size), where each target is a dict containing:
                 "labels": Tensor of dim [num_target_boxes] (where num_target_boxes is the number of ground-truth
                           objects in the target) containing the class labels
                 "boxes": Tensor of dim [num_target_boxes, 4] containing the target box coordinates

        Returns:
            A list of size batch_size, containing tuples of (index_i, index_j) where:
                - index_i is the indices of the selected predictions (in order)
                - index_j is the indices of the corresponding selected targets (in order)
            For each batch element, it holds:
                len(index_i) = len(index_j) = min(num_queries, num_target_boxes)
        """
        bs, num_queries = outputs["pred_logits"].shape[:2]

        # We flatten to compute the cost matrices in a batch
        out_prob = outputs["pred_logits"].flatten(0, 1).softmax(-1)  # [batch_size * num_queries, num_classes]
        out_bbox = outputs["pred_boxes"].flatten(0, 1)  # [batch_size * num_queries, 4]

        # Also concat the target labels and boxes
        tgt_ids = torch.cat([v["labels"] for v in targets])
        tgt_bbox = torch.cat([v["boxes"] for v in targets])

        # Compute the classification cost. Contrary to the loss, we don't use the NLL,
        # but approximate it in 1 - proba[target class].
        # The 1 is a constant that doesn't change the matching, it can be ommitted.
        cost_class = -out_prob[:, tgt_ids]

        # Compute the L1 cost between boxes
        cost_bbox = torch.cdist(out_bbox, tgt_bbox, p=1)

        # Compute the giou cost betwen boxes
        cost_giou = -generalized_box_iou(box_cxcywh_to_xyxy(out_bbox), box_cxcywh_to_xyxy(tgt_bbox))

        # Final cost matrix
        C = self.cost_bbox * cost_bbox + self.cost_class * cost_class + self.cost_giou * cost_giou
        C = C.view(bs, num_queries, -1).cpu()

        sizes = [len(v["boxes"]) for v in targets]
        indices = [linear_sum_assignment(c[i]) for i, c in enumerate(C.split(sizes, -1))]
        return [
            (torch.as_tensor(i, dtype=torch.int64).to("cuda"), torch.as_tensor(j, dtype=torch.int64).to("cuda"))
            for i, j in indices
        ]


def get_src_permutation_idx(indices):
    # permute predictions following indices
    batch_idx = torch.cat([torch.full_like(src, i) for i, (src, _) in enumerate(indices)])
    src_idx = torch.cat([src for (src, _) in indices])
    return batch_idx, src_idx


class APCalculator:
    """A class for calculating average precision (AP).

    This class is built to be used in a training loop, allowing the ground truth (GTs)
    to be initialized once in the __init__ constructor, and then reused many times by calling
    the `calculate_map()` method.

    Attributes:
        iou_threshold (float): The intersection over union (IoU) threshold used for the AP calculation. Defaults to 0.5.
        data_iter (torch.utils.data.DataLoader): A PyTorch dataloader that provides images and targets (e.g., bounding boxes)
            for the dataset.
        n_classes (int): The number of object classes.
        metric (MeanAveragePrecision): An instance of the MeanAveragePrecision class from torchmetrics to compute mAP.

    Args:
        data_iter (torch.utils.data.DataLoader): A PyTorch dataloader that provides images and targets (e.g., bounding boxes)
            for the dataset.
        n_classes (int): The number of object classes.
        iou_threshold (float, optional): The intersection over union (IoU) threshold used for the AP calculation.
            Defaults to 0.5.
    """

    def __init__(self, data_iter):
        """Initializes the APCalculator object with the specified data iterator, number of classes,
        and IoU threshold."""
        self.data_iter = data_iter
        self.metric = MeanAveragePrecision(iou_type="bbox", box_format="cxcywh", class_metrics=True)

    def calculate_map(self, net, nms_threshold=0.1):
        """Calculates the mean average precision (mAP) for the given object detection network.

        Args:
            net (torch.nn.Module): The object detection network.
            nms_threshold (float, optional): The non-maximum suppression (NMS) threshold. Defaults to 0.1.

        Returns:
            dict: A dictionary containing the mAP and other related metrics.
        """
        net.eval()
        for i, (images, targets) in enumerate(self.data_iter):
            preds = []
            GTs = []
            new_targets = []
            for idx in range(targets["labels"].shape[0]):
                labels = targets["labels"][idx]
                boxes = targets["boxes"][idx]
                new_targets.append(
                    {
                        "labels": labels[labels != -1].cpu().detach(),
                        "boxes": boxes[labels != -1].cpu().detach(),
                    }
                )

            for j in range(images.shape[0]):
                GTs.append(
                    {
                        "boxes": new_targets[j]["boxes"],
                        "labels": new_targets[j]["labels"],
                    }
                )
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            images = images.to(device)
            outputs = net(images)
            outputs["pred_logits"] = outputs["pred_logits"].cpu().detach()  # [Bs, N, C]
            outputs["pred_boxes"] = outputs["pred_boxes"].cpu().detach()  # [Bs, N, 4]
            outputs["pred_objectness"] = outputs["pred_objectness"].cpu().detach()  # [Bs, N, 1]
            for j in range(images.shape[0]):
                prob = F.softmax(outputs["pred_logits"][j], dim=1)
                top_p, top_class = prob.topk(1, dim=1)
                boxes = outputs["pred_boxes"][j]
                scores = top_p.squeeze()
                top_class = top_class.squeeze()
                scores = outputs["pred_objectness"][j].squeeze()
                sel_boxes_idx = torchvision.ops.nms(
                    boxes=box_cxcywh_to_xyxy(boxes), scores=scores, iou_threshold=nms_threshold
                )
                preds.append(
                    {
                        "boxes": boxes[sel_boxes_idx],
                        "scores": scores[sel_boxes_idx],
                        "labels": top_class[sel_boxes_idx],
                    }
                )
            self.metric.update(preds, GTs)
        result = self.metric.compute()
        self.metric.reset()
        return result


def plot_bbox(img, boxes, labels):
    """
    Plot bounding boxes on the given image with labels.

    Bounding boxes are defined as tuples (x, y, w, h), where:
        - (x, y) are the center coordinates of the bounding box
        - w is the width of the bounding box
        - h is the height of the bounding box

    The bounding boxes are drawn with a unique color for each label.

    Args:
        img (PIL.Image): The image to plot bounding boxes on.
        boxes (List[Tuple[int, int, int, int]]): A list of bounding boxes.
        labels (List[int]): A list of labels corresponding to the bounding boxes.

    Returns:
        A PIL.Image object representing the original image with the bounding boxes plotted on it.
    """
    draw = ImageDraw.Draw(img)
    colors = ["red", "blue", "green", "yellow", "purple", "orange", "pink", "cyan", "magenta", "lime"]
    for box, label in zip(boxes, labels):
        x, y, w, h = box
        color = colors[label % len(colors)]
        draw.rectangle(
            (x - w / 2, y - h / 2, x + w / 2, y + h / 2),
            outline=color,
            width=3,
        )
        draw.text((x - w / 2, y - h / 2), str(label), fill=color)
    return img


# Well use this function later
def plot_grid(imgs, nrows, ncols):
    """
    This function plots a grid of images using the given list of images.
    The grid has the specified number of rows and columns.
    The size of the figure is set to 10x10 inches.
    Returns None.

    Parameters:
        - imgs (List[PIL.Image]): a list of PIL.Image objects to plot
        - nrows (int): the number of rows in the grid
        - ncols (int): the number of columns in the grid

    Returns:
        None.
    """
    assert len(imgs) == nrows * ncols, "nrows*ncols must be equal to the number of images"
    fig = plt.figure(figsize=(10.0, 10.0))
    grid = ImageGrid(
        fig,
        111,  # similar to subplot(111)
        nrows_ncols=(nrows, ncols),
        axes_pad=0.1,  # pad between axes in inch.
    )
    for ax, im in zip(grid, imgs):
        # Iterating over the grid returns the Axes.
        ax.imshow(im)
    plt.show()


def box_cxcywh_to_xyxy(x):
    """
    Convert bounding boxes from (center x, center y, width, height) format to
    (x1, y1, x2, y2) format.

    Args:
        x (torch.Tensor): A tensor of shape (N, 4) in (center x, center y,
            width, height) format.

    Returns:
        torch.Tensor: A tensor of shape (N, 4) in (x1, y1, x2, y2) format.
    """
    x_c, y_c, w, h = x.unbind(-1)
    x1 = x_c - 0.5 * w
    y1 = y_c - 0.5 * h
    x2 = x_c + 0.5 * w
    y2 = y_c + 0.5 * h
    b = torch.stack([x1, y1, x2, y2], dim=-1)
    return b


def box_xyxy_to_cxcywh(x):
    """
    Convert bounding boxes from (x1, y1, x2, y2) format to (center_x, center_y, width, height)
    format.

    Args:
        x (torch.Tensor): A tensor of shape (N, 4) in (x1, y1, x2, y2) format.

    Returns:
        torch.Tensor: A tensor of shape (N, 4) in (center_x, center_y, width, height) format.
    """
    x0, y0, x1, y1 = x.unbind(-1)
    center_x = (x0 + x1) / 2
    center_y = (y0 + y1) / 2
    width = x1 - x0
    height = y1 - y0
    b = torch.stack([center_x, center_y, width, height], dim=-1)
    return b


def box_xywh_to_xyxy(x):
    """
    Convert bounding box from (x, y, w, h) format to (x1, y1, x2, y2) format.

    Args:
        x (torch.Tensor): A tensor of shape (N, 4) in (x, y, w, h) format.

    Returns:
        torch.Tensor: A tensor of shape (N, 4) in (x1, y1, x2, y2) format.
    """
    x_min, y_min, w, h = x.unbind(-1)
    x_max = x_min + w
    y_max = y_min + h
    b = torch.stack([x_min, y_min, x_max, y_max], dim=-1)
    return b


def predict(model, img, n_classes, nms_threshold=0.1, conf_threshold=0.25):
    model.eval()
    img_size = img.shape[-1]
    outputs = model(img.unsqueeze(0).to("cuda"))
    outputs["pred_logits"] = outputs["pred_logits"].cpu()  # [Bs, N, C]
    outputs["pred_boxes"] = outputs["pred_boxes"].cpu()  # [Bs, N, 4]
    outputs["pred_objectness"] = outputs["pred_objectness"].cpu()  # [Bs, N, 1]
    prob = F.softmax(outputs["pred_logits"][0], dim=1)
    top_p, top_class = prob.topk(1, dim=1)
    boxes = outputs["pred_boxes"][0]
    scores = top_p.squeeze()
    top_class = top_class.squeeze()
    keep = outputs["pred_objectness"][0].squeeze() >= conf_threshold
    boxes = boxes[keep]
    scores = scores[keep]
    top_class = top_class[keep]
    if len(outputs["pred_logits"]) == 0:
        return {
            "boxes": torch.tensor([]),
            "scores": torch.tensor([]),
            "labels": torch.tensor([]),
        }
    sel_boxes_idx = torchvision.ops.nms(boxes=box_cxcywh_to_xyxy(boxes), scores=scores, iou_threshold=nms_threshold)
    return boxes[sel_boxes_idx] * img_size, scores[sel_boxes_idx], top_class[sel_boxes_idx]


class ResizeWithBBox(object):
    """
    Resizes an image and its corresponding bounding boxes.
    """

    def __init__(self, size):
        """
        Initializes the transform.

        Args:
            size: tuple, containing the new size of the image.
        """
        self.size = size

    def __call__(self, image, boxes):
        """
        Applies the transform to an image and its corresponding bounding boxes.

        Args:
            image: PIL.Image object, containing the original image.
            boxes: a list of bounding box coordinates in the format [cx, cy, width, height].

        Returns:
        new_image: PIL.Image object, containing the resized image.
        new_boxes: the bounding box coordinates scaled to the new image size. Range [0-1]
        """

        width_scale = self.size[0] / image.size[0]
        height_scale = self.size[1] / image.size[1]
        new_image = image.resize(self.size)

        new_boxes = []
        for box in boxes:
            x1, y1, w, h = box
            new_x1 = x1 * width_scale
            new_y1 = y1 * height_scale
            new_w = w * width_scale
            new_h = h * height_scale
            new_boxes.append([new_x1 / self.size[0], new_y1 / self.size[1], new_w / self.size[0], new_h / self.size[1]])

        return new_image, new_boxes


class FileBasedAPCalculator:
    """A class for calculating average precision (AP) from text files containing detections.

    This class reads ground truth and prediction bounding boxes from text files and calculates
    the mean average precision (mAP).

    Attributes:
        gt_file (str): Path to the ground truth file.
        pred_file (str): Path to the prediction file.
        metric (MeanAveragePrecision): An instance of MeanAveragePrecision class from torchmetrics.

    Args:
        gt_file (str): Path to the ground truth file with format "file_name, cx, cy, w, h, class_id" per line.
        pred_file (str): Path to the prediction file with format "file_name, cx, cy, w, h, class_id, score" per line.
        box_format (str, optional): Format of the bounding boxes. Defaults to "cxcywh".
    """

    def __init__(self, gt_file, pred_file, box_format="cxcywh"):
        """Initializes the FileBasedAPCalculator object with the ground truth and prediction files."""
        self.gt_file = gt_file
        self.pred_file = pred_file
        self.metric = MeanAveragePrecision(iou_type="bbox", box_format=box_format, class_metrics=True)

    def _parse_file(self, file_path, is_pred=False):
        """Parses a text file containing bounding box information.

        Args:
            file_path (str): Path to the file to parse.
            is_pred (bool, optional): Whether the file contains predictions (with confidence scores).
                                     Defaults to False.

        Returns:
            dict: A dictionary mapping file names to lists of bounding boxes and labels.
        """
        result = {}
        with open(file_path, "r") as f:
            for line in f:
                parts = line.strip().split(",")

                # Skip empty lines or malformed entries
                if len(parts) < 6:
                    continue

                file_name = parts[0].strip()

                # Parse coordinates and convert to float
                cx, cy, w, h = map(float, parts[1:5])
                class_id = int(parts[5])

                # Initialize entry for this file if it doesn't exist
                if file_name not in result:
                    if is_pred:
                        result[file_name] = {"boxes": [], "labels": [], "scores": []}
                    else:
                        result[file_name] = {"boxes": [], "labels": []}

                # Add bounding box and label
                result[file_name]["boxes"].append([cx, cy, w, h])
                result[file_name]["labels"].append(class_id)

                # Add score if it's a prediction file
                if is_pred and len(parts) > 6:
                    score = float(parts[6])
                    result[file_name]["scores"].append(score)

        return result

    def calculate_map(self):
        """Calculates the mean average precision (mAP) using the ground truth and prediction files.

        Returns:
            dict: A dictionary containing the mAP and other related metrics.
        """
        # Parse the ground truth and prediction files
        gt_data = self._parse_file(self.gt_file)
        pred_data = self._parse_file(self.pred_file, is_pred=True)

        # Convert the parsed data to the format expected by the metric
        for file_name in gt_data:
            GTs = []
            preds = []

            # Convert ground truth to tensors
            if file_name in gt_data:
                gt_boxes = torch.tensor(gt_data[file_name]["boxes"], dtype=torch.float32)
                gt_labels = torch.tensor(gt_data[file_name]["labels"], dtype=torch.int64)
                GTs.append({"boxes": gt_boxes, "labels": gt_labels})

            # Convert predictions to tensors if available
            if file_name in pred_data:
                pred_boxes = torch.tensor(pred_data[file_name]["boxes"], dtype=torch.float32)
                pred_labels = torch.tensor(pred_data[file_name]["labels"], dtype=torch.int64)
                pred_scores = torch.tensor(pred_data[file_name]["scores"], dtype=torch.float32)
                preds.append({"boxes": pred_boxes, "scores": pred_scores, "labels": pred_labels})

            # Skip if we don't have both predictions and ground truth
            if not GTs or not preds:
                continue

            # Update the metric
            self.metric.update(preds, GTs)

        # Compute the results
        result = self.metric.compute()
        self.metric.reset()
        return result
