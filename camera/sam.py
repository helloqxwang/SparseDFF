from segment_anything import SamPredictor, sam_model_registry
import numpy as np
import cv2
import matplotlib.pyplot as plt

def show_mask(mask, ax, random_color=False):
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        color = np.array([30/255, 144/255, 255/255, 0.6])
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_image)
    
def show_points(coords, labels, ax, marker_size=375):
    pos_points = coords[labels==1]
    neg_points = coords[labels==0]
    ax.scatter(pos_points[:, 0], pos_points[:, 1], color='green', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)
    ax.scatter(neg_points[:, 0], neg_points[:, 1], color='red', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)   
    
def show_box(box, ax):
    x0, y0 = box[0], box[1]
    w, h = box[2] - box[0], box[3] - box[1]
    ax.add_patch(plt.Rectangle((x0, y0), w, h, edgecolor='green', facecolor=(0,0,0,0), lw=2))  

def vis_mask_image(image, mask, points, labels, bbox=None, save_path="./test.png"):
    plt.figure(figsize=(10,10))
    plt.imshow(image)
    show_mask(mask, plt.gca())
    if points is not None:
        show_points(points, labels, plt.gca())
    if bbox is not None:
        show_box(bbox, plt.gca())
    plt.axis('off')
    plt.savefig(save_path)

class Sam_Detector():
    def __init__(self, sam_checkpoint = "./thirdparty_module/sam_vit_h_4b8939.pth", 
                 model_type = "vit_h", device = "cuda") -> None:

        sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
        sam.to(device=device)
        self.predictor = SamPredictor(sam)
    
    def refine_mask(self, mask_ori, input_point, input_label, step_add_num=2, opt_step:int=3):

        mask = mask_ori
        for i in range(opt_step):
            posi_index = np.nonzero(mask)
            posi_select_id = np.random.choice(posi_index[0].shape[0], step_add_num, replace=False)
            posi_select_id[0] = np.argmin(posi_index[0])
            posi_select_id[1] = np.argmax(posi_index[0])
            posi_select_id[2] = np.argmin(posi_index[1])
            posi_select_id[3] = np.argmax(posi_index[1])
            extra_positives = np.array([[posi_index[1][i], posi_index[0][i]] for i in posi_select_id])
            input_point_ = np.concatenate([input_point, extra_positives], axis=0)
            input_label_ = np.concatenate([input_label, np.ones(step_add_num)], axis=0)
            masks, scores, logits = self.predictor.predict(
                point_coords=input_point_,
                point_labels=input_label_,
                multimask_output=True,
                box=None,
            )
            # area = [np.sum(mask) for mask in masks]
            mask = masks[np.argmax(scores)]
            input_point = input_point_
            input_label = input_label_
        return mask

    def get_mask(self, image:np.ndarray, points:np.ndarray, labels:np.ndarray, bbox:np.ndarray=None, secondary_num:int=4):
        """get the mask of the object using both the point and the bounding box

        
        - the input_point is a point near the center of the table.(Object must on there)
        - the bounding box will be an area above the table
        - a negative point is a point outside the bounding box which counts


        Args:
            image (np.ndarray): (h, w, 3)
            points (np.ndarray): (n, 2)
            labels (np.ndarray): (n, )
            bbox (np.ndarray): (4, )
        
        Returns:
            mask (np.ndarray): (h, w)

        """
        input_point = points
        input_label = labels
        input_bbox = bbox
        self.predictor.set_image(image)
        masks, scores, logits = self.predictor.predict(
            point_coords=input_point,
            point_labels=input_label,
            multimask_output=True,
            box=input_bbox,
        )
        area = [np.sum(mask) for mask in masks]
        mask = masks[np.argmax(area)]
        mask = self.refine_mask(mask, input_point, input_label, step_add_num=4, opt_step=4)
        return mask
        
if __name__ == "__main__":
    detector  = Sam_Detector()
    image = cv2.imread("/home/user/wangqx/stanford/kinect/data/20230913_191328/000262413912/colors.png")
    h, w = image.shape[:2]
    bbox = np.array([20, 20, w-30, h-30])
    points = np.array([[w//2, h//2], [200, 200]])
    labels = np.array([1, 0])
    mask= detector.get_mask(image, points, labels, bbox)
    plt.figure(figsize=(10,10))
    plt.imshow(image)
    show_mask(mask, plt.gca())
    show_points(points, labels, plt.gca())
    show_box(bbox, plt.gca())
    plt.axis('off')
    plt.savefig("test.png")