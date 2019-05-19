import rospy
from styx_msgs.msg import TrafficLight
import tensorflow as tf
import numpy as np
from PIL import Image
from PIL import ImageColor
from scipy.stats import norm
import cv2

class TLClassifier(object):
    def __init__(self):
        #TODO load classifier
        #pass
        SSD_GRAPH_FILE = 'ssd_mobilenet_v1_coco_11_06_2017/frozen_inference_graph.pb'
        graph = tf.Graph()
        with graph.as_default():
            od_graph_def = tf.GraphDef()
            with tf.gfile.GFile(SSD_GRAPH_FILE, 'rb') as fid:
                serialized_graph = fid.read()
                od_graph_def.ParseFromString(serialized_graph)
                tf.import_graph_def(od_graph_def, name = '')
        self.detection_graph = graph
        self.image_tensor = self.detection_graph.get_tensor_by_name('image_tensor:0')
        self.detection_boxes = self.detection_graph.get_tensor_by_name('detection_boxes:0')
        self.detection_scores = self.detection_graph.get_tensor_by_name('detection_scores:0')
        self.detection_classes = self.detection_graph.get_tensor_by_name('detection_classes:0')
        self.box_coords = None
        self.height = None
        self.width = None

    def get_classification(self, image):
        """Determines the color of the traffic light in the image

        Args:
            image (cv::Mat): image containing the traffic light

        Returns:
            int: ID of traffic light color (specified in styx_msgs/TrafficLight)

        """
        #TODO implement light color prediction
        #return TrafficLight.UNKNOWN
	# convert to PIL image
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = Image.fromarray(image)

        image_np = np.expand_dims(np.asarray(image, dtype=np.uint8),0)
        with tf.Session(graph=self.detection_graph) as sess:
            (boxes, scores, classes) = sess.run([self.detection_boxes, self.detection_scores, self.detection_classes],
                                                feed_dict={self.image_tensor: image_np})
            boxes = np.squeeze(boxes)
            scores = np.squeeze(scores)
            classes = np.squeeze(classes)
            
            confidence_cutoff = 0.2
            boxes, scores,classes = self.filter_boxes(confidence_cutoff, boxes, scores, classes)
            self.width, self.height = image.size
            self.box_coords = self.to_image_coords(boxes, self.height, self.width)
            
        rospy.logwarn("--->Light image#: {0}".format(len(classes)))
        if len(classes) == 0:
            return TrafficLight.UNKNOWN
        # save the pixel number to judge the color
        th = [120, 180]                 # threshold
        px_num = [0, 0, 0]
        for k in range(len(classes)):
            if classes[k] == 10:            # traffice light ID
                #rospy.logwarn("--->Light: {0}, {1}".format(len(classes), self.box_coords[k,:]))
                #box_light = (int(self.box_coords[k,1]),int(self.height-self.box_coords[k,2]),int(self.box_coords[k,3]),int(self.height-self.box_coords[k,0]))
                box_light = (int(self.box_coords[k,1]),int(self.box_coords[k,0]),int(self.box_coords[k,3]),int(self.box_coords[k,2]))
                image_light = image.crop(box_light)
                mr = np.zeros(image_light.size)
                my = np.zeros(image_light.size)
                mg = np.zeros(image_light.size)
                px = image_light.load()
                for i in range(image_light.size[0]):
                    for j in range(image_light.size[1]):
                        if px[i,j][0]>th[1] and px[i,j][1]<th[0] and px[i,j][2]<th[0]:
                            mr[i,j] += 1
                        if px[i,j][0]>th[1] and px[i,j][1]>th[1] and px[i,j][2]<th[0]:
                            my[i,j] += 1
                        if px[i,j][0]<th[0] and px[i,j][1]>th[1] and px[i,j][2]<th[0]:
                            mg[i,j] += 1
                px_num[0] += np.sum(mr)
                px_num[1] += np.sum(my) 
                px_num[2] += np.sum(mg)
        #rospy.logwarn("--->Light: {0}, {1}".format(len(classes), boxes_coords))
        if np.argmax(px_num) == 0:
            return TrafficLight.RED
        elif np.argmax(px_num) == 1:
            return TrafficLight.YELLOW
        elif np.argmax(px_num) == 2:
            return TrafficLight.GREEN

    def filter_boxes(self, min_score, boxes, scores, classes):
        n = len(classes)
        idxs = []
        for i in range(n):
            if scores[i] >= min_score:
                idxs.append(i)
        filtered_boxes = boxes[idxs, ...]
        filtered_scores = scores[idxs, ...]
        filtered_classes = classes[idxs, ...]
        return filtered_boxes, filtered_scores, filtered_classes

    def to_image_coords(self, boxes, height, width):
        box_coords = np.zeros_like(boxes)
        box_coords[:,0] = boxes[:,0] * height
        box_coords[:,1] = boxes[:,1] * width
        box_coords[:,2] = boxes[:,2] * height
        box_coords[:,3] = boxes[:,3] * width
        return box_coords
        
