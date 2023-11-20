import cv2
import numpy as np

def find_object_center(image_path1, image_path2, center_coord, box_size=50):
    # 读取图片
    image_A = cv2.imread(image_path1, cv2.IMREAD_COLOR)
    image_B = cv2.imread(image_path2, cv2.IMREAD_COLOR)

    # 定义目标区域
    x, y = center_coord
    x1, y1 = max(x - box_size // 2, 0), max(y - box_size // 2, 0)
    x2, y2 = min(x + box_size // 2, image_A.shape[1]), min(y + box_size // 2, image_A.shape[0])
    object_roi = image_A[y1:y2, x1:x2]

    # 使用SIFT特征检测器
    feature_detector = cv2.SIFT_create()

    # 在两个图像中检测特征
    keypoints_A, descriptors_A = feature_detector.detectAndCompute(object_roi, None)
    keypoints_B, descriptors_B = feature_detector.detectAndCompute(image_B, None)

    # 使用FLANN匹配器
    FLANN_INDEX_KDTREE = 1
    index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
    search_params = dict(checks = 50)
    flann = cv2.FlannBasedMatcher(index_params, search_params)
    matches = flann.knnMatch(descriptors_A, descriptors_B, k=2)

    # Lowe's Ratio Test
    good_matches = []
    for m, n in matches:
        if m.distance < 0.7 * n.distance:
            good_matches.append(m)

    # 计算匹配点的均值位置
    if len(good_matches) > 10:  # 选择一个合适的匹配阈值
        points = np.zeros((len(good_matches), 2), dtype=np.float32)
        for i, match in enumerate(good_matches):
            points[i, :] = keypoints_B[match.trainIdx].pt

        mean_point = np.mean(points, axis=0)
        return tuple(mean_point.astype(int))
    else:
        return None

# 示例用法
image_path1 = '1.jpg'
image_path2 = '2.jpg'
center_coord = (287, 683)  # 用实际坐标替换这些值

center_point = find_object_center(image_path1, image_path2, center_coord)
print("Center Point in Image 2:", center_point)
