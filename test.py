import tensorflow as tf
import numpy as np
import cv2
import os
import tools
import matplotlib.pyplot as plt
from network import Yolov3


def video_show(frame, boxes, scores, labels, classes):
	color_list = [[0, 0, 255], [255, 0, 0], [0, 255, 0], [0, 255, 255], [203, 192, 255]]
	for i in range(len(labels)):
		xmin, ymin, xmax, ymax = boxes[i]
		xmin, ymin = min(max(xmin, 0), 1280), min(max(ymin, 0), 720)
		xmax, ymax = max(min(xmax, 1280), 0), max(min(ymax, 720), 0)
		if xmax - xmin >10 and ymax - ymin > 10:
			score, category = scores[i], classes[labels[i]]
			cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), color_list[labels[i]], 2)
			cv2.putText(frame, "%s: %.2f" % (category, score), (xmin, ymin+15), cv2.FONT_HERSHEY_COMPLEX, 0.8, (0, 255, 0), 2)
	return frame


save_path = "D:/program/avp_yolo/video.avi"
ckpt_path = "D:/program/avp_yolo/model1/model0.ckpt"
classes = ["alien", "corpse", "person", "flare"]
network = Yolov3(class_num=len(classes))
ori_height, ori_width, new_height, new_width = 720, 1280, 416, 416
scale = np.max((ori_height / new_height, ori_width / new_width))
new_h, new_w = int(ori_height / scale), int(ori_width / scale)

config = tf.compat.v1.ConfigProto()
config.allow_soft_placement = True
config.gpu_options.allow_growth = True

with tf.compat.v1.Session(config=config) as sess:
	input_data = tf.placeholder(tf.float32, [new_height, new_width, 3], name="input_data")
	with tf.compat.v1.variable_scope("yolov3"):
		predict_feature_maps = network.forward(tf.expand_dims(input_data, axis=0), is_training=False)
	pred_boxes, pred_confs, pred_probs = network.predict(predict_feature_maps)
	pred_scores = pred_confs * pred_probs
	boxes, scores, labels = tools.gpu_nms(pred_boxes, pred_scores, len(classes), max_boxes=200, score_thresh=0.3, nms_thresh=0.45)
	saver = tf.compat.v1.train.Saver()
	sess.run([tf.compat.v1.global_variables_initializer(), tf.compat.v1.local_variables_initializer()])
	saver.restore(sess, ckpt_path)
	print("Load network: " + ckpt_path)

	videoCapture = cv2.VideoCapture("E:/FFOutput/demo.mp4")
	fps = videoCapture.get(cv2.CAP_PROP_FPS)
	size = (int(videoCapture.get(cv2.CAP_PROP_FRAME_WIDTH)), int(videoCapture.get(cv2.CAP_PROP_FRAME_HEIGHT)))
	#videoWriter = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc('I', '4', '2', '0'), fps, size)
	cv2.namedWindow("001")
	res, frame = videoCapture.read()
	while res and cv2.waitKey(1) != 27:
		image = cv2.cvtColor(cv2.resize(tools.bgr_balance(frame), (new_w, new_h)), cv2.COLOR_BGR2RGB).astype(np.float32) / 255.
		image = cv2.copyMakeBorder(image, 0, new_height - new_h, 0, new_width - new_w, cv2.BORDER_CONSTANT, value=0)
		boxes_, scores_, labels_ = sess.run([boxes, scores, labels], feed_dict={input_data: image})
		#boxes_, scores_, labels_ = tools.filte_bbox((boxes_ * scale).astype(np.int64), scores_, labels_)
		frame = video_show(frame, (np.array(boxes_) * scale).astype(np.int64), scores_, labels_, classes)
		cv2.imshow("001", frame)
		#videoWriter.write(frame)
		res, frame = videoCapture.read()
	cv2.destroyWindow("001")
	videoCapture.release()