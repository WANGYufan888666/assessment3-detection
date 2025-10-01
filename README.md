# assessment3-detection
import jetson.inference
import jetson.utils

# 加载模型
net = jetson.inference.detectNet("ssd-mobilenet-v2", threshold=0.5)

# 加载图片
image1_path = "/home/nvidia/Desktop/1.png"  # 水果篮
image2_path = "/home/nvidia/Desktop/2.png"  # 猫和狗
image1 = jetson.utils.loadImage(image1_path)
image2 = jetson.utils.loadImage(image2_path)

# 检测图片中的对象
detections1 = net.Detect(image1)
detections2 = net.Detect(image2)

# 打印检测结果
print("Image 1 Results:")
for detection in detections1:
    print(detection)

print("\nImage 2 Results:")
for detection in detections2:
    print(detection)

# 渲染检测框并保存结果图片
output1_path = "/home/nvidia/Desktop/output_image1.jpg"
output2_path = "/home/nvidia/Desktop/output_image2.jpg"

jetson.utils.saveImage(output1_path, image1)
jetson.utils.saveImage(output2_path, image2) 
