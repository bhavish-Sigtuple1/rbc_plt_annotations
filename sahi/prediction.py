import copy
from typing import Dict, List, Optional, Union
import numpy as np
from PIL import Image
import cv2


def get_bool_mask_from_coco_segmentation(coco_segmentation, width, height):
    """
    Convert coco segmentation to 2D boolean mask of given height and width
    """
    size = [height, width]
    points = [np.array(point).reshape(-1, 2).round().astype(int) for point in coco_segmentation]
    bool_mask = np.zeros(size)
    bool_mask = cv2.fillPoly(bool_mask, points, 1)
    bool_mask.astype(bool)
    return bool_mask

class PredictionScore:
    def __init__(self, value: float):
        """
        Arguments:
            score: prediction score between 0 and 1
        """
        # if score is a numpy object, convert it to python variable
        if type(value).__module__ == "numpy":
            value = copy.deepcopy(value).tolist()
        # set score
        self.value = value

    def is_greater_than_threshold(self, threshold):
        """
        Check if score is greater than threshold
        """
        return self.value > threshold

    def __repr__(self):
        return f"PredictionScore: <value: {self.value}>"


class BoundingBox:
    """
    Bounding box of the annotation.
    """

    def __init__(self, box: List[float], shift_amount: List[int] = [0, 0]):
        """
        Args:
            box: List[float]
                [minx, miny, maxx, maxy]
            shift_amount: List[int]
                To shift the box and mask predictions from sliced image
                to full sized image, should be in the form of [shift_x, shift_y]
        """
        if box[0] < 0 or box[1] < 0 or box[2] < 0 or box[3] < 0:
            raise Exception("Box coords [minx, miny, maxx, maxy] cannot be negative")
        self.minx = box[0]
        self.miny = box[1]
        self.maxx = box[2]
        self.maxy = box[3]

        self.shift_x = shift_amount[0]
        self.shift_y = shift_amount[1]

    @property
    def shift_amount(self):
        """
        Returns the shift amount of the bbox slice as [shift_x, shift_y]
        """
        return [self.shift_x, self.shift_y]

    @property
    def area(self):
        return (self.maxx - self.minx) * (self.maxy - self.miny)

    def get_expanded_box(self, ratio=0.1, max_x=None, max_y=None):
        w = self.maxx - self.minx
        h = self.maxy - self.miny
        y_mar = int(w * ratio)
        x_mar = int(h * ratio)
        maxx = min(max_x, self.maxx + x_mar) if max_x else self.maxx + x_mar
        minx = max(0, self.minx - x_mar)
        maxy = min(max_y, self.maxy + y_mar) if max_y else self.maxy + y_mar
        miny = max(0, self.miny - y_mar)
        box = [minx, miny, maxx, maxy]
        return BoundingBox(box)

    def to_xywh(self):
        """
        Returns: [xmin, ymin, width, height]
        """
        return [self.minx, self.miny, self.maxx - self.minx, self.maxy - self.miny]

    def to_coco_bbox(self):
        """
        Returns: [xmin, ymin, width, height]
        """
        return self.to_xywh()

    def to_xyxy(self):
        """
        Returns: [xmin, ymin, xmax, ymax]
        """
        return [self.minx, self.miny, self.maxx, self.maxy]

    def to_voc_bbox(self):
        """
        Returns: [xmin, ymin, xmax, ymax]
        """
        return self.to_xyxy()

    def get_shifted_box(self):
        """
        Returns: shifted BoundingBox
        """
        box = [
            self.minx + self.shift_x,
            self.miny + self.shift_y,
            self.maxx + self.shift_x,
            self.maxy + self.shift_y,
        ]
        return BoundingBox(box)

    def __repr__(self):
        return f"BoundingBox: <{(self.minx, self.miny, self.maxx, self.maxy)}, w: {self.maxx - self.minx}, h: {self.maxy - self.miny}>"


class Category:
    """
    Category of the annotation.
    """

    def __init__(self, id=None, name=None):
        """
        Args:
            id: int
                ID of the object category
            name: str
                Name of the object category
        """
        if not isinstance(id, int):
            raise TypeError("id should be integer")
        if not isinstance(name, str):
            raise TypeError("name should be string")
        self.id = id
        self.name = name

    def __repr__(self):
        return f"Category: <id: {self.id}, name: {self.name}>"



class ObjectAnnotation:
    """
    All about an annotation such as Mask, Category, BoundingBox.
    """

    @classmethod
    def from_bool_mask(
        cls,
        bool_mask,
        category_id: Optional[int] = None,
        category_name: Optional[str] = None,
        shift_amount: Optional[List[int]] = [0, 0],
        full_shape: Optional[List[int]] = None,
    ):
        """
        Creates ObjectAnnotation from bool_mask (2D np.ndarray)

        Args:
            bool_mask: np.ndarray with bool elements
                2D mask of object, should have a shape of height*width
            category_id: int
                ID of the object category
            category_name: str
                Name of the object category
            full_shape: List
                Size of the full image, should be in the form of [height, width]
            shift_amount: List
                To shift the box and mask predictions from sliced image to full
                sized image, should be in the form of [shift_x, shift_y]
        """
        return cls(
            category_id=category_id,
            bool_mask=bool_mask,
            category_name=category_name,
            shift_amount=shift_amount,
            full_shape=full_shape,
        )

    @classmethod
    def from_coco_segmentation(
        cls,
        segmentation,
        full_shape: List[int],
        category_id: Optional[int] = None,
        category_name: Optional[str] = None,
        shift_amount: Optional[List[int]] = [0, 0],
    ):
        """
        Creates ObjectAnnotation from coco segmentation:
        [
            [x1, y1, x2, y2, x3, y3, ...],
            [x1, y1, x2, y2, x3, y3, ...],
            ...
        ]

        Args:
            segmentation: List[List]
                [
                    [x1, y1, x2, y2, x3, y3, ...],
                    [x1, y1, x2, y2, x3, y3, ...],
                    ...
                ]
            category_id: int
                ID of the object category
            category_name: str
                Name of the object category
            full_shape: List
                Size of the full image, should be in the form of [height, width]
            shift_amount: List
                To shift the box and mask predictions from sliced image to full
                sized image, should be in the form of [shift_x, shift_y]
        """
        bool_mask = get_bool_mask_from_coco_segmentation(segmentation, width=full_shape[1], height=full_shape[0])
        return cls(
            category_id=category_id,
            bool_mask=bool_mask,
            category_name=category_name,
            shift_amount=shift_amount,
            full_shape=full_shape,
        )

    @classmethod
    def from_coco_bbox(
        cls,
        bbox: List[int],
        category_id: Optional[int] = None,
        category_name: Optional[str] = None,
        shift_amount: Optional[List[int]] = [0, 0],
        full_shape: Optional[List[int]] = None,
    ):
        """
        Creates ObjectAnnotation from coco bbox [minx, miny, width, height]

        Args:
            bbox: List
                [minx, miny, width, height]
            category_id: int
                ID of the object category
            category_name: str
                Name of the object category
            full_shape: List
                Size of the full image, should be in the form of [height, width]
            shift_amount: List
                To shift the box and mask predictions from sliced image to full
                sized image, should be in the form of [shift_x, shift_y]
        """
        xmin = bbox[0]
        ymin = bbox[1]
        xmax = bbox[0] + bbox[2]
        ymax = bbox[1] + bbox[3]
        bbox = [xmin, ymin, xmax, ymax]
        return cls(
            category_id=category_id,
            bbox=bbox,
            category_name=category_name,
            shift_amount=shift_amount,
            full_shape=full_shape,
        )

    @classmethod
    def from_coco_annotation_dict(
        cls,
        annotation_dict: Dict,
        full_shape: List[int],
        category_name: str = None,
        shift_amount: Optional[List[int]] = [0, 0],
    ):
        """
        Creates ObjectAnnotation object from category name and COCO formatted
        annotation dict (with fields "bbox", "segmentation", "category_id").

        Args:
            annotation_dict: dict
                COCO formatted annotation dict (with fields "bbox", "segmentation", "category_id")
            category_name: str
                Category name of the annotation
            full_shape: List
                Size of the full image, should be in the form of [height, width]
            shift_amount: List
                To shift the box and mask predictions from sliced image to full
                sized image, should be in the form of [shift_x, shift_y]
        """
        if annotation_dict["segmentation"]:
            return cls.from_coco_segmentation(
                segmentation=annotation_dict["segmentation"],
                category_id=annotation_dict["category_id"],
                category_name=category_name,
                shift_amount=shift_amount,
                full_shape=full_shape,
            )
        else:
            return cls.from_coco_bbox(
                bbox=annotation_dict["bbox"],
                category_id=annotation_dict["category_id"],
                category_name=category_name,
                shift_amount=shift_amount,
                full_shape=full_shape,
            )

    @classmethod
    def from_shapely_annotation(
        cls,
        annotation,
        full_shape: List[int],
        category_id: Optional[int] = None,
        category_name: Optional[str] = None,
        shift_amount: Optional[List[int]] = [0, 0],
    ):
        """
        Creates ObjectAnnotation from shapely_utils.ShapelyAnnotation

        Args:
            annotation: shapely_utils.ShapelyAnnotation
            category_id: int
                ID of the object category
            category_name: str
                Name of the object category
            full_shape: List
                Size of the full image, should be in the form of [height, width]
            shift_amount: List
                To shift the box and mask predictions from sliced image to full
                sized image, should be in the form of [shift_x, shift_y]
        """
        bool_mask = get_bool_mask_from_coco_segmentation(
            annotation.to_coco_segmentation(), width=full_shape[1], height=full_shape[0]
        )
        return cls(
            category_id=category_id,
            bool_mask=bool_mask,
            category_name=category_name,
            shift_amount=shift_amount,
            full_shape=full_shape,
        )

    @classmethod
    def from_imantics_annotation(
        cls,
        annotation,
        shift_amount: Optional[List[int]] = [0, 0],
        full_shape: Optional[List[int]] = None,
    ):
        """
        Creates ObjectAnnotation from imantics.annotation.Annotation

        Args:
            annotation: imantics.annotation.Annotation
            shift_amount: List
                To shift the box and mask predictions from sliced image to full
                sized image, should be in the form of [shift_x, shift_y]
            full_shape: List
                Size of the full image, should be in the form of [height, width]
        """
        return cls(
            category_id=annotation.category.id,
            bool_mask=annotation.mask.array,
            category_name=annotation.category.name,
            shift_amount=shift_amount,
            full_shape=full_shape,
        )

    def __init__(
        self,
        bbox: Optional[List[int]] = None,
        bool_mask: Optional[np.ndarray] = None,
        category_id: Optional[int] = None,
        category_name: Optional[str] = None,
        shift_amount: Optional[List[int]] = [0, 0],
        full_shape: Optional[List[int]] = None,
    ):
        """
        Args:
            bbox: List
                [minx, miny, maxx, maxy]
            bool_mask: np.ndarray with bool elements
                2D mask of object, should have a shape of height*width
            category_id: int
                ID of the object category
            category_name: str
                Name of the object category
            shift_amount: List
                To shift the box and mask predictions from sliced image
                to full sized image, should be in the form of [shift_x, shift_y]
            full_shape: List
                Size of the full image after shifting, should be in
                the form of [height, width]
        """
        if not isinstance(category_id, int):
            raise ValueError("category_id must be an integer")
        if (bbox is None) and (bool_mask is None):
            raise ValueError("you must provide a bbox or bool_mask")

       
        self.mask = None

        # if bbox is a numpy object, convert it to python List[float]
        if type(bbox).__module__ == "numpy":
            bbox = copy.deepcopy(bbox).tolist()

        # make sure bbox coords lie inside [0, image_size]
        xmin = max(bbox[0], 0)
        ymin = max(bbox[1], 0)
        if full_shape:
            xmax = min(bbox[2], full_shape[1])
            ymax = min(bbox[3], full_shape[0])
        else:
            xmax = bbox[2]
            ymax = bbox[3]
        bbox = [xmin, ymin, xmax, ymax]
        # set bbox
        self.bbox = BoundingBox(bbox, shift_amount)

        category_name = category_name if category_name else str(category_id)
        self.category = Category(
            id=category_id,
            name=category_name,
        )

        self.merged = None

    def deepcopy(self):
        """
        Returns: deepcopy of current ObjectAnnotation instance
        """
        return copy.deepcopy(self)

    def get_shifted_object_annotation(self):
        if self.mask:
            return ObjectAnnotation(
                bbox=self.bbox.get_shifted_box().to_xyxy(),
                category_id=self.category.id,
                bool_mask=self.mask.get_shifted_mask().bool_mask,
                category_name=self.category.name,
                shift_amount=[0, 0],
                full_shape=self.mask.get_shifted_mask().full_shape,
            )
        else:
            return ObjectAnnotation(
                bbox=self.bbox.get_shifted_box().to_xyxy(),
                category_id=self.category.id,
                bool_mask=None,
                category_name=self.category.name,
                shift_amount=[0, 0],
                full_shape=None,
            )

    def __repr__(self):
        return f"""ObjectAnnotation<
    bbox: {self.bbox},
    mask: {self.mask},
    category: {self.category}>"""

class ObjectPrediction(ObjectAnnotation):
    """
    Class for handling detection model predictions.
    """

    def __init__(
        self,
        bbox: Optional[List[int]] = None,
        category_id: Optional[int] = None,
        category_name: Optional[str] = None,
        bool_mask: Optional[np.ndarray] = None,
        score: Optional[float] = 0,
        shift_amount: Optional[List[int]] = [0, 0],
        full_shape: Optional[List[int]] = None,
    ):
        """
        Creates ObjectPrediction from bbox, score, category_id, category_name, bool_mask.

        Arguments:
            bbox: list
                [minx, miny, maxx, maxy]
            score: float
                Prediction score between 0 and 1
            category_id: int
                ID of the object category
            category_name: str
                Name of the object category
            bool_mask: np.ndarray
                2D boolean mask array. Should be None if model doesn't output segmentation mask.
            shift_amount: list
                To shift the box and mask predictions from sliced image
                to full sized image, should be in the form of [shift_x, shift_y]
            full_shape: list
                Size of the full image after shifting, should be in
                the form of [height, width]
        """
        self.score = PredictionScore(score)
        super().__init__(
            bbox=bbox,
            category_id=category_id,
            bool_mask=bool_mask,
            category_name=category_name,
            shift_amount=shift_amount,
            full_shape=full_shape,
        )

    def get_shifted_object_prediction(self):
        """
        Returns shifted version ObjectPrediction.
        Shifts bbox and mask coords.
        Used for mapping sliced predictions over full image.
        """
        if self.mask:
            return ObjectPrediction(
                bbox=self.bbox.get_shifted_box().to_xyxy(),
                category_id=self.category.id,
                score=self.score.value,
                bool_mask=self.mask.get_shifted_mask().bool_mask,
                category_name=self.category.name,
                shift_amount=[0, 0],
                full_shape=self.mask.get_shifted_mask().full_shape,
            )
        else:
            return ObjectPrediction(
                bbox=self.bbox.get_shifted_box().to_xyxy(),
                category_id=self.category.id,
                score=self.score.value,
                bool_mask=None,
                category_name=self.category.name,
                shift_amount=[0, 0],
                full_shape=None,
            )

    def __repr__(self):
        return f"""ObjectPrediction<
    bbox: {self.bbox},
    mask: {self.mask},
    score: {self.score},
    category: {self.category}>"""


def read_image_as_pil(image: Union[Image.Image, str, np.ndarray], exif_fix: bool = False):
    Image.MAX_IMAGE_PIXELS = None

    if isinstance(image, Image.Image):
        image_pil = image
    elif isinstance(image, np.ndarray):
        if image.shape[0] < 5:  # image in CHW
            image = image[:, :, ::-1]
        image_pil = Image.fromarray(image)
    else:
        raise TypeError("read image with 'pillow' using 'Image.open()'")
    return image_pil


class PredictionResult:
    def __init__(
        self,
        object_prediction_list: List[ObjectPrediction],
        image: Union[Image.Image, str, np.ndarray],
        durations_in_seconds: Optional[Dict] = None,
    ):
        self.image: List[Image.Image] =  [read_image_as_pil(img) for img in image]
        self.image_width_list, self.image_height_list = [img.size[0] for img in self.image], [img.size[1] for img in self.image]
        self.object_prediction_list: List[ObjectPrediction] = object_prediction_list
        self.durations_in_seconds = durations_in_seconds