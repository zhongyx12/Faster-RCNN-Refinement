import cv2
import cPickle, os
import numpy as np
from faster_rcnn.datasets.factory import get_imdb

# up to 5
det_file_dirs = ['/home/yang/Faster-RCNN-Refinement/output/faster_rcnn_end2end/voc_2007_test/faster_rcnn_100000']
visualization_output_dir = 'visualization'
# orange, yellow, green, blue, red
colors = [(66, 164, 244), (65, 235, 244), (0, 204, 0), (244, 181, 65), (65, 82, 244)]

def vis_detections(im, class_name, dets, model_idx, thresh=0.8, text=False):
    """Visual debugging of detections."""
    for i in range(np.minimum(10, dets.shape[0])):
        bbox = tuple(int(np.round(x)) for x in dets[i, :4])
        score = dets[i, -1]
        if score > thresh:
            cv2.rectangle(im, bbox[0:2], bbox[2:4], colors[model_idx], 2)
            if text:
                cv2.putText(im, '%s: %.3f' % (class_name, score), (bbox[0], bbox[1] + 15), cv2.FONT_HERSHEY_PLAIN,
                            1.0, (0, 0, 255), thickness=1)
    return im


def visualize(imdb):
    """Test a Fast R-CNN network on an image database."""
    num_images = len(imdb.image_index)
    # all detections are collected into:
    #    all_boxes[cls][image] = N x 5 array of detections in
    #    (x1, y1, x2, y2, score)
    detection_results = []
    for det_file_dir in det_file_dirs:
        det_file = os.path.join(det_file_dir, 'detections.pkl')
        with open(det_file, 'rb') as f:
            all_boxes = cPickle.load(f)
        detection_results.append(all_boxes)

    for i in range(num_images):
        im = cv2.imread(imdb.image_path_at(i))
        im2show = np.copy(im)

        for model_idx, detection_result in enumerate(detection_results):
            # skip j = 0, because it's the background class
            for j in xrange(1, imdb.num_classes):
                cls_dets = detection_result[j][i]
                im2show = vis_detections(im2show, imdb.classes[j], cls_dets, model_idx)

        cv2.imwrite(os.path.join(visualization_output_dir, str(i) + '.png'), im2show)
        if (i+1) % 100 == 0:
            print('im_write: %d/%d' % (i+1, num_images))


if __name__ == '__main__':
    imdb_name = 'voc_2007_test'
    imdb = get_imdb(imdb_name)
    imdb.competition_mode(on=True)
    visualize(imdb)
