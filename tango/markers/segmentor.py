import math
import cv2
import numpy as np
import torch
import supervision as sv
from PIL import Image
from transformers import pipeline
from typing import Optional, Tuple

from tango.markers.postprocessing import masks_to_marks


class SegmentAnythingMarkGenerator:
    """
    A class for performing image segmentation using a specified model.

    Parameters:
        device (str): The device to run the model on (e.g., 'cpu', 'cuda').
        model_name (str): The name of the model to be loaded. Defaults to
            'facebook/sam-vit-huge'.
    """
    def __init__(self, device: str = 'cpu', model_name: str = "facebook/sam-vit-huge"):
        from transformers import SamModel, SamProcessor, SamImageProcessor
        self.model = SamModel.from_pretrained(model_name).to(device)
        self.processor = SamProcessor.from_pretrained(model_name)
        self.image_processor = SamImageProcessor.from_pretrained(model_name)
        self.device = device
        self.pipeline = pipeline(
            task="mask-generation",
            model=self.model,
            image_processor=self.image_processor,
            device=self.device)
        # Cache slot populated by get_roi_feature() on first call.
        self._cached_embeddings: Optional[torch.Tensor] = None  # [1, 256, 64, 64]

    @torch.no_grad()
    def generate(
        self,
        image: Image.Image,
        mask: Optional[np.ndarray] = None
    ) -> sv.Detections:
        """
        Generate image segmentation marks.

        Parameters:
            image (np.ndarray): The image to be marked in BGR format.
            mask: (Optional[np.ndarray]): The mask to be used as a guide for
                segmentation.

        Returns:
            sv.Detections: An object containing the segmentation masks and their
                corresponding bounding box coordinates.
        """
        #image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        if mask is None:
            outputs = self.pipeline(image, points_per_batch=64)
            masks = np.array(outputs['masks'])
            return masks_to_marks(masks=masks)
        else:
            inputs = self.processor(image, return_tensors="pt").to(self.device)
            image_embeddings = self.model.get_image_embeddings(inputs.pixel_values)
            masks = []
            for polygon in sv.mask_to_polygons(mask.astype(bool)):
                indexes = np.random.choice(a=polygon.shape[0], size=5, replace=True)
                input_points = polygon[indexes]
                inputs = self.processor(
                    images=image,
                    input_points=[[input_points]],
                    return_tensors="pt"
                ).to(self.device)
                del inputs["pixel_values"]
                outputs = self.model(image_embeddings=image_embeddings, **inputs)
                mask = self.processor.image_processor.post_process_masks(
                    masks=outputs.pred_masks.cpu().detach(),
                    original_sizes=inputs["original_sizes"].cpu().detach(),
                    reshaped_input_sizes=inputs["reshaped_input_sizes"].cpu().detach()
                )[0][0][0].numpy()
                masks.append(mask)
            masks = np.array(masks)
            return masks_to_marks(masks=masks)

    @torch.no_grad()
    def get_roi_feature(
        self,
        bbox: Tuple[int, int, int, int],
        image: Optional[Image.Image] = None,
    ) -> np.ndarray:
        """Return a 256-dim ROI feature vector from SAM's ViT-H image encoder.

        Args:
            bbox: (x1, y1, x2, y2) bounding box in original image pixel coords.
            image: Source image used to populate the embedding cache on first
                   call.  Ignored once the cache is warm.  Required when the
                   cache is empty (i.e. generate() has not been called yet).

        Returns:
            np.ndarray of shape (256,) — mean-pooled feature over the ROI.

        Note:
            SAM preprocesses images to 1024×1024 before encoding, giving a
            64×64 feature map (stride 16 in 1024-px space).  The pixel→cell
            mapping here uses stride=16 directly on the *original* pixel coords,
            which is exact when the input is already 1024×1024 and a close
            approximation for other sizes.
        """
        if self._cached_embeddings is None:
            if image is None:
                raise ValueError(
                    "No cached image embeddings. Pass image= to populate "
                    "the cache on the first call."
                )
            inputs = self.processor(image, return_tensors="pt").to(self.device)
            self._cached_embeddings = self.model.get_image_embeddings(
                inputs.pixel_values
            )  # [1, 256, 64, 64]

        feat = self._cached_embeddings          # [1, 256, 64, 64]
        _, C, H, W = feat.shape                 # C=256, H=W=64

        STRIDE = 16
        x1, y1, x2, y2 = bbox
        fx1 = max(0, math.floor(x1 / STRIDE))
        fy1 = max(0, math.floor(y1 / STRIDE))
        fx2 = min(W, max(fx1 + 1, math.ceil(x2 / STRIDE)))
        fy2 = min(H, max(fy1 + 1, math.ceil(y2 / STRIDE)))

        roi = feat[0, :, fy1:fy2, fx1:fx2]     # [256, h, w]
        vec = roi.mean(dim=(-2, -1))            # [256]
        return vec.cpu().numpy()                # np.ndarray (256,)
