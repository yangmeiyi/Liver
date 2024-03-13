import os

def read_boxes_from_txt(file_path):
    with open(file_path, 'r') as file:
        lines = file.readlines()
        boxes = [list(map(float, line.strip().split())) for line in lines]  # Assuming format: class xmin ymin xmax ymax
    return boxes

def calculate_metrics(true_boxes, pred_boxes, iou_threshold=0.5):
    true_positive = 0
    false_positive = 0
    false_negative = len(true_boxes)
    for true_box in true_boxes:
        true_class, true_xmin, true_ymin, true_xmax, true_ymax = true_box
        true_box_coords = [true_xmin, true_ymin, true_xmax, true_ymax] 
        
        for pred_box in pred_boxes: 
            
            pred_class, pred_confidence, pred_xmin, pred_ymin, pred_xmax, pred_ymax = pred_box
            pred_box_coords = [pred_xmin, pred_ymin, pred_xmax, pred_ymax]

            intersection = max(0, min(pred_box_coords[2], true_box_coords[2]) - max(pred_box_coords[0], true_box_coords[0])) * \
                           max(0, min(pred_box_coords[3], true_box_coords[3]) - max(pred_box_coords[1], true_box_coords[1]))
            union = (pred_box_coords[2] - pred_box_coords[0]) * (pred_box_coords[3] - pred_box_coords[1]) + \
                    (true_box_coords[2] - true_box_coords[0]) * (true_box_coords[3] - true_box_coords[1]) - intersection
            iou = intersection / union if union > 0 else 0

            if iou >= iou_threshold and pred_class == true_class:
                true_positive += 1

                break



    return true_positive,len(pred_boxes),len(true_boxes)

def main():
    true_boxes_dir = "detection/mAP-master/input/ground-truth/"
    pred_boxes_dir = "detection/mAP-master/input/detection-results/"

    image_files = os.listdir(true_boxes_dir)
    total_tp=0
    total_p=0
    total_t=0
    for image_file in image_files:
        true_boxes_path = os.path.join(true_boxes_dir, image_file)
        pred_boxes_path = os.path.join(pred_boxes_dir, image_file)

        true_boxes = read_boxes_from_txt(true_boxes_path)
        pred_boxes = read_boxes_from_txt(pred_boxes_path)

        tp, fp, fn = calculate_metrics(true_boxes, pred_boxes)
        total_tp+=tp
        total_p+=fp
        total_t+=fn
    recall=total_tp/total_t
    precision=total_tp/total_p
    f1=2*recall*precision/(recall+precision)
    print(f"True Positive: {f1}, False Positive: {recall}, False Negative: {precision}")

if __name__ == "__main__":
    main()
