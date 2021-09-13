import tensorflow as tf
import numpy as np
import cv2
import tensorflow.contrib.slim as slim


def bgr_balance(image):
	channels = cv2.split(image)
	clahe = cv2.createCLAHE(clipLimit=5, tileGridSize=(8, 8))
	clahe.apply(channels[0], channels[0])
	clahe.apply(channels[1], channels[1])
	clahe.apply(channels[2], channels[2])
	return cv2.merge(channels)

def hsv_balance(image):
	image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
	channels = cv2.split(image)
	clahe = cv2.createCLAHE(clipLimit=5, tileGridSize=(8, 8))
	clahe.apply(channels[2], channels[2])
	image = cv2.cvtColor(cv2.merge(channels), cv2.COLOR_HSV2BGR)
	return image

def yuv_balance(image):
	image = cv2.cvtColor(image, cv2.COLOR_BGR2YCrCb)
	channels = cv2.split(image)
	clahe = cv2.createCLAHE(clipLimit=5, tileGridSize=(8, 8))
	clahe.apply(channels[0], channels[0])
	clahe.apply(channels[1], channels[1])
	clahe.apply(channels[2], channels[2])
	image = cv2.cvtColor(cv2.merge(channels), cv2.COLOR_YCrCb2BGR)
	return image

def filte_bbox(boxes, scores, labels):
	the_list, boxes_out, scores_out, labels_out = [], [], [], []
	for i in range(len(labels)):
		xmin, ymin, xmax, ymax = np.array(boxes[i]).astype(np.int64)
		width, height = xmax - xmin, ymax - ymin
		if width > 10 and height > 10:
			the_list.append(i)
	for i in the_list:
		boxes_out.append(boxes[i])
		scores_out.append(scores[i])
		labels_out.append(labels[i])
	return boxes_out, scores_out, labels_out

def conv2d(inputs, filters, kernel_size, strides = 1):
	def _fixed_padding(inputs, kernel_size):
		pad_total = kernel_size - 1
		pad_beg = pad_total // 2
		pad_end = pad_total - pad_beg
		padded_inputs = tf.pad(inputs, [[0, 0], [pad_beg, pad_end], [pad_beg, pad_end], [0, 0]])
		return padded_inputs
	if strides > 1:
		inputs = _fixed_padding(inputs, kernel_size)
	inputs = slim.conv2d(inputs, filters, kernel_size, stride=strides, padding = "SAME" if strides == 1 else "VALID")
	return inputs

def darknet53_body(inputs):
	def res_block(inputs, filters):
		shortcut = inputs
		net = conv2d(inputs, filters = filters * 1, kernel_size = 1)
		net = conv2d(net, filters = filters * 2, kernel_size = 3)
		net = net + shortcut
		return net
	net = conv2d(inputs, filters = 32, kernel_size = 3)
	net = conv2d(net, filters = 64, kernel_size = 3, strides = 2)
	net = res_block(net, filters = 32)
	net = conv2d(net, filters = 128, kernel_size = 3, strides = 2)
	for _ in range(2):
		net = res_block(net, filters = 64)
	net = conv2d(net, filters = 256, kernel_size = 3, strides = 2)
	for _ in range(8):
		net = res_block(net, filters = 128)
	route_1 = net
	net = conv2d(net, filters = 512, kernel_size = 3, strides = 2)
	for _ in range(8):
		net = res_block(net, filters = 256)
	route_2 = net
	net = conv2d(net, filters = 1024, kernel_size = 3, strides = 2)
	for _ in range(4):
		net = res_block(net, filters = 512)
	route_3 = net
	return route_1, route_2, route_3

def yolo_block(inputs, filters):
	net = conv2d(inputs, filters = filters * 1, kernel_size = 1)
	net = conv2d(net, filters = filters * 2, kernel_size = 3)
	net = conv2d(net, filters = filters * 1, kernel_size = 1)
	net = conv2d(net, filters = filters * 2, kernel_size = 3)
	net = conv2d(net, filters = filters * 1, kernel_size = 1)
	route = net
	net = conv2d(net, filters = filters * 2, kernel_size = 3)
	return route, net

def unsample_layer(inputs, out_shape):
	inputs = tf.compat.v1.image.resize_nearest_neighbor(inputs, [out_shape[1], out_shape[2]], name = "upsampled")
	return inputs

def gpu_nms(boxes, scores, num_classes, max_boxes=50, score_thresh=0.5, nms_thresh=0.5):
	boxes_list, label_list, score_list = [], [], []
	max_boxes = tf.constant(max_boxes, dtype=tf.int32)
	boxes, scores = tf.reshape(boxes, [-1, 4]), tf.reshape(scores, [-1, num_classes])
	mask = tf.math.greater_equal(scores, tf.constant(score_thresh))
	for i in range(num_classes):
		filter_boxes = tf.boolean_mask(boxes, mask[:, i])
		filter_scores = tf.boolean_mask(scores[:, i], mask[:, i])
		nms_indices = tf.image.non_max_suppression(boxes=filter_boxes, scores=filter_scores, max_output_size=max_boxes,
			iou_threshold=nms_thresh, name="nms_indices")
		label_list.append(tf.ones_like(tf.gather(filter_scores, nms_indices), tf.int32) * i)
		boxes_list.append(tf.gather(filter_boxes, nms_indices))
		score_list.append(tf.gather(filter_scores, nms_indices))
	boxes, label, score = tf.concat(boxes_list, axis=0), tf.concat(label_list, axis=0), tf.concat(score_list, axis=0)
	return boxes, score, label