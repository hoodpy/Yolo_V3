import tensorflow as tf
import tensorflow.contrib.slim as slim
import numpy as np
import tools
import os
import cv2
import time
import xml.etree.ElementTree as ET


class Timer():
	def __init__(self):
		self._total_time = 0
		self._calls = 0
		self._start_time = 0
		self._diff = 0
		self._average_time = 0

	def tic(self):
		self._start_time = time.time()

	def toc(self, average=True):
		self._diff = time.time() - self._start_time
		self._total_time += self._diff
		self._calls += 1
		self._average_time = self._total_time / self._calls


class Yolov3():
	def __init__(self, class_num, static_shape=True, focal_loss=False):
		self._anchors = np.asarray([[10,13],[16,30],[33,23],[30,61],[62,45],[59,119],[116,90],[156,198],[373,326]], np.float32)
		self._class_num = class_num
		self._static_shape = static_shape
		self._focal_loss = focal_loss

	def forward(self, inputs, is_training=False, reuse=False):
		self._image_size = tf.shape(inputs)[1:3]
		batch_norm_params = {"decay": 0.99, "epsilon": 1e-5, "scale": True, "is_training": is_training, "fused": None}
		with slim.arg_scope([slim.conv2d, slim.batch_norm], reuse=reuse):
			with slim.arg_scope([slim.conv2d], normalizer_fn=slim.batch_norm, normalizer_params=batch_norm_params,
				biases_initializer=None, activation_fn=lambda x: tf.nn.leaky_relu(x, alpha=0.1)):
				with tf.compat.v1.variable_scope("darknet53_body"):
					route_1, route_2, route_3 = tools.darknet53_body(inputs)

				with tf.compat.v1.variable_scope("yolov3_head"):
					inter1, net = tools.yolo_block(route_3, filters=512)
					feature_map_1 = slim.conv2d(net, 3 * (5 + self._class_num), [1, 1], stride=1, normalizer_fn=None,
						activation_fn=None, biases_initializer=tf.zeros_initializer())
					feature_map_1 = tf.identity(feature_map_1, name="feature_map_1")

					inter1 = tools.conv2d(inter1, filters=256, kernel_size=1)
					inter1 = tools.unsample_layer(inter1, out_shape=route_2.get_shape().as_list() if self._static_shape else 
						tf.shape(route_2))
					concat1 = tf.concat([inter1, route_2], axis=3)

					inter2, net = tools.yolo_block(concat1, filters=256)
					feature_map_2 = slim.conv2d(net, 3 * (5 + self._class_num), [1, 1], stride=1, normalizer_fn=None, 
						activation_fn=None, biases_initializer=tf.zeros_initializer())
					feature_map_2 = tf.identity(feature_map_2, name="feature_map_2")

					inter2 = tools.conv2d(inter2, filters=128, kernel_size=1)
					inter2 = tools.unsample_layer(inter2, out_shape=route_1.get_shape().as_list() if self._static_shape else 
						tf.shape(route_1))
					concat2 = tf.concat([inter2, route_1], axis=3)

					_, feature_map_3 = tools.yolo_block(concat2, filters=128)
					feature_map_3 = slim.conv2d(feature_map_3, 3 * (5 + self._class_num), [1, 1], stride=1, normalizer_fn=None, 
						activation_fn=None, biases_initializer=tf.zeros_initializer())
					feature_map_3 = tf.identity(feature_map_3, name="feature_map_3")

		return feature_map_1, feature_map_2, feature_map_3

	def reorg_layer(self, feature_map, anchors):
		grid_size = feature_map.get_shape().as_list()[1:3] if self._static_shape else tf.shape(feature_map)[1:3]
		ratio = tf.cast(self._image_size / grid_size, tf.float32)
		rescaled_anchors = [(anchor[0] / ratio[1], anchor[1] / ratio[0]) for anchor in anchors]
		feature_map = tf.reshape(feature_map, [-1, grid_size[0], grid_size[1], 3, 5 + self._class_num])
		box_centers, box_sizes, conf_logits, prob_logits = tf.split(feature_map, [2, 2, 1, self._class_num], axis=-1)
		box_centers = tf.nn.sigmoid(box_centers)
		grid_x, grid_y = tf.range(grid_size[1], dtype=tf.int32), tf.range(grid_size[0], dtype=tf.int32)
		grid_x, grid_y = tf.meshgrid(grid_x, grid_y)
		x_offset, y_offset = tf.reshape(grid_x, [-1, 1]), tf.reshape(grid_y, [-1, 1])
		x_y_offset = tf.concat([x_offset, y_offset], axis=-1)
		x_y_offset = tf.cast(tf.reshape(x_y_offset, [grid_size[0], grid_size[1], 1, 2]), tf.float32)
		box_centers = (box_centers + x_y_offset) * ratio[::-1]
		box_sizes = (tf.math.exp(box_sizes) * rescaled_anchors) * ratio[::-1]
		boxes = tf.concat([box_centers, box_sizes], axis=-1)
		return x_y_offset, boxes, conf_logits, prob_logits

	def predict(self, feature_maps):
		feature_map_1, feature_map_2, feature_map_3 = feature_maps
		feature_map_anchors = [(feature_map_1, self._anchors[6:9]), (feature_map_2, self._anchors[3:6]), 
			(feature_map_3, self._anchors[:3])]
		recog_results = [self.reorg_layer(feature_map, anchor) for (feature_map, anchor) in feature_map_anchors]

		def _reshape(result):
			x_y_offset, boxes, conf_logits, prob_logits = result
			grid_size = x_y_offset.get_shape().as_list()[:2] if self._static_shape else tf.shape(x_y_offset)[:2]
			boxes = tf.reshape(boxes, [-1, grid_size[0] * grid_size[1] * 3, 4])
			conf_logits = tf.reshape(conf_logits, [-1, grid_size[0] * grid_size[1] * 3, 1])
			prob_logits = tf.reshape(prob_logits, [-1, grid_size[0] * grid_size[1] * 3, self._class_num])
			return boxes, conf_logits, prob_logits

		boxes_list, confs_list, probs_list = [], [], []
		for result in recog_results:
			boxes, conf_logits, prob_logits = _reshape(result)
			confs, probs = tf.sigmoid(conf_logits), tf.sigmoid(prob_logits)
			boxes_list.append(boxes)
			confs_list.append(confs)
			probs_list.append(probs)

		boxes, confs, probs = tf.concat(boxes_list, axis=1), tf.concat(confs_list, axis=1), tf.concat(probs_list, axis=1)
		center_x, center_y, width, height = tf.split(boxes, [1, 1, 1, 1], axis=-1)
		x_min, y_min = center_x - width / 2, center_y - height / 2
		x_max, y_max = center_x + width / 2, center_y + height / 2
		boxes = tf.concat([x_min, y_min, x_max, y_max], axis=-1)
		return boxes, confs, probs

	def loss_layer(self, feature_map_i, y_true, anchors, pos_weights):
		grid_size = tf.shape(feature_map_i)[1:3]
		ratio = tf.cast(self._image_size / grid_size, tf.float32)
		N = tf.cast(tf.shape(feature_map_i)[0], tf.float32)
		x_y_offset, pred_boxes, pred_conf_logits, pred_prob_logits = self.reorg_layer(feature_map_i, anchors)
		object_mask = y_true[..., 4:5]
		ignore_mask = tf.TensorArray(tf.float32, size=0, dynamic_size=True)

		def loop_cond(idx, ignore_mask):
			return tf.math.less(idx, tf.cast(N, tf.int32))

		def loop_body(idx, ignore_mask):
			valid_true_boxes = tf.boolean_mask(y_true[idx, ..., 0:4], tf.cast(object_mask[idx, ..., 0], "bool"))
			iou = self.box_iou(pred_boxes[idx], valid_true_boxes)
			best_iou = tf.math.reduce_max(iou, axis=-1)
			ignore_mask_tmp = tf.cast(best_iou < 0.5, tf.float32)
			ignore_mask = ignore_mask.write(idx, ignore_mask_tmp)
			return idx + 1, ignore_mask

		_, ignore_mask = tf.while_loop(cond=loop_cond, body=loop_body, loop_vars=[0, ignore_mask])
		ignore_mask = tf.expand_dims(ignore_mask.stack(), -1)
		pred_box_xy, pred_box_wh = pred_boxes[..., 0:2], pred_boxes[..., 2:4]
		true_xy, pred_xy = y_true[..., 0:2] / ratio[::-1] - x_y_offset, pred_box_xy / ratio[::-1] - x_y_offset
		true_tw_th, pred_tw_th = y_true[..., 2:4] /anchors, pred_box_wh / anchors
		true_tw_th = tf.where_v2(condition=tf.math.equal(true_tw_th, 0), x=tf.ones_like(true_tw_th), y=true_tw_th)
		pred_tw_th = tf.where_v2(condition=tf.math.equal(pred_tw_th, 0), x=tf.ones_like(pred_tw_th), y=pred_tw_th)
		true_tw_th = tf.math.log(tf.clip_by_value(true_tw_th, 1e-9, 1e9))
		pred_tw_th = tf.math.log(tf.clip_by_value(pred_tw_th, 1e-9, 1e9))
		box_loss_scale = 2.0 - (y_true[..., 2:3] / tf.cast(self._image_size[1], 
			tf.float32)) * (y_true[..., 3:4] / tf.cast(self._image_size[0], tf.float32))
		mix_w = y_true[..., -1:]
		xy_loss = tf.math.reduce_sum(tf.math.square(true_xy - pred_xy) * object_mask * box_loss_scale * mix_w) / N
		wh_loss = tf.math.reduce_sum(tf.math.square(true_tw_th - pred_tw_th) * object_mask * box_loss_scale * mix_w) / N

		conf_pos_mask = object_mask
		conf_neg_mask = (1 - object_mask) * ignore_mask
		conf_loss_pos = conf_pos_mask * tf.nn.sigmoid_cross_entropy_with_logits(labels=object_mask, logits=pred_conf_logits)
		conf_loss_neg = conf_neg_mask * tf.nn.sigmoid_cross_entropy_with_logits(labels=object_mask, logits=pred_conf_logits)
		conf_loss = conf_loss_pos + conf_loss_neg
		if self._focal_loss:
			alpha, gamma = 1.0, 2.0
			focal_mask = alpha * tf.math.pow(tf.math.abs(object_mask - tf.math.sigmoid(pred_conf_logits)), gamma)
			conf_loss *= focal_mask
		conf_loss = tf.math.reduce_sum(conf_loss * mix_w) / N

		class_loss = object_mask * tf.nn.weighted_cross_entropy_with_logits(labels=y_true[..., 5:-1], logits=pred_prob_logits,
			pos_weight=pos_weights) * mix_w
		#class_loss = object_mask * tf.nn.sigmoid_cross_entropy_with_logits(labels=y_true[..., 5:-1], 
		#	logits=pred_prob_logits) * mix_w
		class_loss = tf.math.reduce_sum(class_loss) / N
		
		return xy_loss, wh_loss, conf_loss, class_loss

	def box_iou(self, pred_boxes, valid_true_boxes):
		pred_box_xy, pred_box_wh = pred_boxes[..., 0:2], pred_boxes[..., 2:4]
		pred_box_xy, pred_box_wh = tf.expand_dims(pred_box_xy, -2), tf.expand_dims(pred_box_wh, -2)
		true_box_xy, true_box_wh = valid_true_boxes[..., 0:2], valid_true_boxes[..., 2:4]
		intersect_mins = tf.math.maximum(pred_box_xy - pred_box_wh / 2., true_box_xy - true_box_wh / 2.)
		intersect_maxs = tf.math.minimum(pred_box_xy + pred_box_wh / 2., true_box_xy + true_box_wh / 2.)
		intersect_wh = tf.math.maximum(intersect_maxs - intersect_mins, 0.)
		intersect_area = intersect_wh[..., 0] * intersect_wh[..., 1]
		pred_box_area = pred_box_wh[..., 0] * pred_box_wh[..., 1]
		true_box_area = true_box_wh[..., 0] * true_box_wh[..., 1]
		true_box_area = tf.expand_dims(true_box_area, axis=0)
		iou = intersect_area / (pred_box_area + true_box_area - intersect_area + 1e-10)
		return iou

	def compute_loss(self, y_pred, y_true, pos_weights):
		loss_xy, loss_wh, loss_conf, loss_class = 0., 0., 0., 0.
		anchor_group = [self._anchors[6:9], self._anchors[3:6], self._anchors[0:3]]
		for i in range(len(y_pred)):
			result = self.loss_layer(y_pred[i], y_true[i], anchor_group[i], pos_weights)
			loss_xy += result[0]
			loss_wh += result[1]
			loss_conf += result[2]
			loss_class += result[3]
		total_loss = loss_xy + loss_wh + loss_conf + loss_class
		return [total_loss, loss_xy, loss_wh, loss_conf, loss_class]


class Train():
	def __init__(self):
		self._train_file = "D:/program/avp_yolo/train.txt"
		self._xml_path1 = "E:/labeling/data_xml/"
		self._xml_path2 = "E:/labeling/new_data_xml/"
		self._image_path2 = "E:/labeling/new_data_picture/"
		self._ckpt_path = "D:/program/garbage_yolo/data/darknet_weights/yolov3.ckpt"
		self._log_path = "D:/program/avp_yolo/log/"
		self._save_path = "D:/program/avp_yolo/model/"
		self._classes = ["alien", "corpse", "person", "flare"]
		self._pos_weights = np.array([3.0, 1.0, 1.0, 1.0])
		self._class_num = len(self._classes)
		self._batch_size = 5
		self._image_size = [416, 416]
		self._learning_rate = 1e-4
		self._network = Yolov3(class_num=self._class_num, static_shape=False, focal_loss=True)
		self._anchors = self._network._anchors
		self._timer = Timer()

	def parse_line(self, line):
		tree = ET.parse(line)
		root = tree.getroot()
		pic_path = r'' + root.find("path").text
		img_width = int(root.find("size").find("width").text)
		img_height = int(root.find("size").find("height").text)
		boxes, labels = [], []
		for instance in root.findall("object"):
			x_min = float(instance.find("bndbox").find("xmin").text)
			y_min = float(instance.find("bndbox").find("ymin").text)
			x_max = float(instance.find("bndbox").find("xmax").text)
			y_max = float(instance.find("bndbox").find("ymax").text)
			boxes.append([x_min, y_min, x_max, y_max])
			labels.append(int(self._classes.index(instance.find("name").text)))
		boxes, labels = np.asarray(boxes), np.asarray(labels)
		return pic_path, boxes, labels, img_width, img_height

	def resize_with_bbox(self, pic_path, bbox, ori_width, ori_height, new_width, new_height):
		img = cv2.resize(tools.bgr_balance(cv2.imread(pic_path)), (new_width, new_height))
		bbox[:, [0, 2]] = bbox[:, [0, 2]] * new_width / ori_width
		bbox[:, [1, 3]] = bbox[:, [1, 3]] * new_height / ori_height
		return img, bbox

	def process_box(self, boxes, labels):
		anchors_mask = [[6, 7, 8], [3, 4, 5], [0, 1, 2]]
		box_centers = (boxes[:, 0:2] + boxes[:, 2:4]) / 2
		box_sizes = boxes[:, 2:4] - boxes[:, 0:2]
		y_true_13 = np.zeros((self._image_size[0] // 32, self._image_size[1] // 32, 3, 6 + self._class_num), np.float32)
		y_true_26 = np.zeros((self._image_size[0] // 16, self._image_size[1] // 16, 3, 6 + self._class_num), np.float32)
		y_true_52 = np.zeros((self._image_size[0] // 8, self._image_size[1] // 8, 3, 6 + self._class_num), np.float32)
		y_true_13[..., -1], y_true_26[..., -1], y_true_52[..., -1] = 1., 1., 1.
		y_true = [y_true_13, y_true_26, y_true_52]
		box_sizes = np.expand_dims(box_sizes, 1)
		mins = np.maximum(- box_sizes / 2, - self._anchors / 2)
		maxs = np.minimum(box_sizes / 2, self._anchors / 2)
		whs = maxs - mins
		iou = (whs[:, :, 0] * whs[:, :, 1]) / (box_sizes[:, :, 0] * box_sizes[:, :, 1] + 
			self._anchors[:, 0] * self._anchors[:, 1]-whs[:, :, 0] * whs[:, :, 1] + 1e-10)
		best_match_idx = np.argmax(iou, axis=1)
		ratio_dict = {1: 8., 2.: 16., 3.: 32.}
		for i, idx in enumerate(best_match_idx):
			feature_map_group, ratio = 2 - idx // 3, ratio_dict[np.ceil((idx + 1) / 3.)]
			x, y = int(np.floor(box_centers[i, 0] / ratio)), int(np.floor(box_centers[i, 1] / ratio))
			k, c = anchors_mask[feature_map_group].index(idx), labels[i]
			y_true[feature_map_group][y, x, k, :2] = box_centers[i]
			y_true[feature_map_group][y, x, k, 2:4] = box_sizes[i]
			y_true[feature_map_group][y, x, k, 4] = 1.
			y_true[feature_map_group][y, x, k, 5 + c] = 1.
			y_true[feature_map_group][y, x, k, -1] = boxes[i, -1]
		return y_true_13, y_true_26, y_true_52

	def parse_data(self, line, letterbox_resize):
		pic_path, boxes, labels, img_width, img_height = self.parse_line(line)
		boxes = np.concatenate((boxes, np.full(shape=(boxes.shape[0], 1), fill_value=1., dtype=np.float32)), axis=-1)
		img, boxes = self.resize_with_bbox(pic_path, boxes, img_width, img_height, self._image_size[1], self._image_size[0])
		img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.
		y_true_13, y_true_26, y_true_52 = self.process_box(boxes, labels)
		return img, y_true_13, y_true_26, y_true_52

	def get_batch_data(self, batch_line, letterbox_resize=True):
		img_batch, y_true_13_batch, y_true_26_batch, y_true_52_batch = [], [], [], []
		batch_line = batch_line.tolist()
		for line in batch_line:
			img, y_true_13, y_true_26, y_true_52 = self.parse_data(line, letterbox_resize)
			img_batch.append(img)
			y_true_13_batch.append(y_true_13)
			y_true_26_batch.append(y_true_26)
			y_true_52_batch.append(y_true_52)
		img_batch, y_true_13_batch = np.asarray(img_batch), np.asarray(y_true_13_batch)
		y_true_26_batch, y_true_52_batch = np.asarray(y_true_26_batch), np.asarray(y_true_52_batch)
		return img_batch, y_true_13_batch, y_true_26_batch, y_true_52_batch

	def gen_train_file(self):
		names1 = [self._xml_path1 + name for name in os.listdir(self._xml_path1)]
		names2 = [self._xml_path2 + name for name in os.listdir(self._xml_path2)]
		names = names1 + names2
		content = ""
		for name in names:
			content += name + "\n"
		with open(self._train_file, "a+") as f:
			f.write(content)
		return len(names)

	def get_dataset(self, epochs):
		self._samples_num = self.gen_train_file()
		dataset = tf.data.TextLineDataset(self._train_file)
		dataset = dataset.shuffle(self._samples_num).repeat(epochs).batch(self._batch_size)
		dataset = dataset.map(lambda x: tf.py_func(self.get_batch_data, [x], [tf.float32, tf.float32, 
			tf.float32, tf.float32]), num_parallel_calls=10)
		dataset = dataset.prefetch(5)
		self._iterator = dataset.make_initializable_iterator()
		images, y_true_13, y_true_26, y_true_52 = self._iterator.get_next()
		return images, y_true_13, y_true_26, y_true_52

	def train(self, index, epochs):
		config = tf.compat.v1.ConfigProto()
		config.allow_soft_placement = True
		config.gpu_options.allow_growth = True
		with tf.compat.v1.Session(config=config) as sess:
			global_step = tf.Variable(0, trainable=False)
			#learning_rate = tf.Variable(self._learning_rate, trainable=False)
			#tf.compat.v1.summary.scalar("learning_rate", learning_rate)

			images, y_true_13, y_true_26, y_true_52 = self.get_dataset(epochs)
			images.set_shape([None, None, None, 3])
			y_true_13.set_shape([None, None, None, None, None])
			y_true_26.set_shape([None, None, None, None, None])
			y_true_52.set_shape([None, None, None, None, None])

			decay_steps = int(self._samples_num * epochs / self._batch_size / (epochs / 10))
			learning_rate = tf.compat.v1.train.polynomial_decay(self._learning_rate, global_step, decay_steps, 
				self._learning_rate * 0.1, power=0.5, cycle=True, name="learning_rate")
			tf.compat.v1.summary.scalar("learning_rate", learning_rate)

			with tf.compat.v1.variable_scope("yolov3"):
				pred_features_maps = self._network.forward(images, is_training=True)
			loss = self._network.compute_loss(pred_features_maps, [y_true_13, y_true_26, y_true_52], self._pos_weights)

			#if index > 0:
			#	saver_to_restore = tf.compat.v1.train.Saver(var_list=tf.contrib.framework.get_variables_to_restore(include=None,
			#		exclude=["Variable", "Variable_1"]))
			saver_to_restore = tf.compat.v1.train.Saver(var_list=tf.contrib.framework.get_variables_to_restore(include=None, 
				exclude=["yolov3/yolov3_head/Conv_14", "yolov3/yolov3_head/Conv_6", "yolov3/yolov3_head/Conv_22", 
				"Variable", "learning_rate"]))
			tf.compat.v1.summary.scalar("train_batch_statistics/total_loss", loss[0])
			tf.compat.v1.summary.scalar("train_batch_statistics/loss_xy", loss[1])
			tf.compat.v1.summary.scalar("train_batch_statistics/loss_wh", loss[2])
			tf.compat.v1.summary.scalar("train_batch_statistics/loss_conf", loss[3])
			tf.compat.v1.summary.scalar("train_batch_statistics/loss_class", loss[4])

			update_ops = tf.compat.v1.get_collection(tf.compat.v1.GraphKeys.UPDATE_OPS)
			optimizer = tf.compat.v1.train.AdamOptimizer(learning_rate)
			grades_and_vars = optimizer.compute_gradients(loss[0], var_list=tf.contrib.framework.get_variables_to_restore(
				include=["yolov3/yolov3_head"]))
			with tf.control_dependencies(update_ops):
				#train_op = tf.compat.v1.train.AdamOptimizer(learning_rate).minimize(loss[0], global_step=global_step)
				train_op = optimizer.apply_gradients(grades_and_vars, global_step=global_step)

			self._saver = tf.compat.v1.train.Saver(var_list=tf.compat.v1.global_variables(), max_to_keep=5)
			merged = tf.compat.v1.summary.merge_all()
			summary_writer = tf.compat.v1.summary.FileWriter(self._log_path, sess.graph)

			sess.run([tf.compat.v1.global_variables_initializer(), tf.compat.v1.local_variables_initializer()])
			#if index > 0:
			#	saver_to_restore.restore(sess, self._save_path + "model" + str(index-1) + ".ckpt")
			#	print("Load network: " + self._save_path + "model" + str(index-1) + ".ckpt")
			saver_to_restore.restore(sess, self._ckpt_path)
			print("Load network: " + self._ckpt_path)
			sess.run(self._iterator.initializer)

			while True:
				try:
					self._timer.tic()
					_, summary, _loss, steps = sess.run([train_op, merged, loss, global_step])
					summary_writer.add_summary(summary, global_step=steps)
					self._timer.toc()
					#if (steps + 1) == int(self._samples_num * epochs / self._batch_size * 0.8):
					#	sess.run(tf.compat.v1.assign(learning_rate, self._learning_rate * 0.1))
					if (steps + 1) % 300 == 0:
						print(">>>Steps: %d\n>>>Total_loss: %.6f\n>>>Loss_xy: %.6fs\n>>>Loss_wh: %.6f\n>>>Loss_conf: %.6f" % (
							steps + 1, _loss[0], _loss[1], _loss[2], _loss[3]))
						print(">>>Loss_class: %.6f\n>>>Average_time: %.6f\n" % (_loss[4], self._timer._average_time))
				except tf.errors.OutOfRangeError:
					break

			self.snap_shot(sess, index)
			for name in os.listdir(self._xml_path2):
				os.remove(self._xml_path2 + name)
			for name in os.listdir(self._image_path2):
				os.remove(self._image_path2 + name)
			os.remove(self._train_file)
			print("Remove other xmls and images.\n")

	def snap_shot(self, sess, iter):
		network = self._network
		file_name = self._save_path + "model" + str(iter) + ".ckpt"
		self._saver.save(sess, file_name)
		print("Wrote snapshot to: " + file_name + "\n")


if __name__ == "__main__":
	trainer = Train()
	trainer.train(index=0, epochs=100)