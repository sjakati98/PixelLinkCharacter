# IoU

IoU is defined as the Intersection Over Union. The purpose of this metric is to determine the average IoU of the predicted bounding boxes against the ground truth bounding boxes.

Formulation of the metrics score are according to the following rules:
* any predicted annotation with an IoU score of `0.5` or higher is considered a true positive with that ground truth annotation
* any predicted annotation with no IoU score higher than `0.5` compared to any ground truth annotation is considered a false positive
* any ground truth annotation with no associated true positive predictions is considered a false negative