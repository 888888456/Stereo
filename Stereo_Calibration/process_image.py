import os
import cv2

# 原始图像路径
image_path = "Stereo_Raw/"

# 分割图像保存的路径
save_image = "left_right_image/"

# 确保保存目录存在
if not os.path.exists(save_image):
    os.makedirs(save_image)

# 获取图像列表
image_list = os.listdir(image_path)

i = 0
for img_name in image_list:
    # 读取原始图像
    raw_image = cv2.imread(os.path.join(image_path, img_name))

    # 检查图像是否成功读取
    if raw_image is None:
        print(f"警告: 无法读取图像 {img_name}，跳过")
        continue

    # 先裁掉上下左右各 1 像素
    # raw_image[h, w] -> raw_image[1:h-1, 1:w-1]
    h, w = raw_image.shape[:2]
    if h <= 2 or w <= 2:
        print(f"图像过小无法裁剪: {img_name}，跳过")
        continue
    raw_image = raw_image[1:h-1, 1:w-1]

    # 更新裁剪后的尺寸
    height, width = raw_image.shape[:2]

    # 计算中点位置
    mid_point = width // 2

    # 分割左右图像
    left_image = raw_image[:, 0:mid_point]
    right_image = raw_image[:, mid_point:width]

    # 保存左右图像
    image_left_name = os.path.join(save_image, f"left{i}.jpg")
    image_right_name = os.path.join(save_image, f"right{i}.jpg")

    cv2.imwrite(image_left_name, left_image)
    cv2.imwrite(image_right_name, right_image)

    print(f"已处理: {img_name} (原始: {w}x{h}, 裁剪后: {width}x{height})")
    i += 1

print(f"\n处理完成！共处理 {i} 张图像")
