import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.models import load_model

# 加載 CIFAR-10 資料集
(_, _), (x_test, y_test) = cifar10.load_data()

# 資料預處理（與訓練時一致）
x_test = x_test.astype('float32') / 255.0

# CIFAR-10 類別標籤
class_labels = [
    'Airplane', 'Automobile', 'Bird', 'Cat', 'Deer',
    'Dog', 'Frog', 'Horse', 'Ship', 'Truck'
]

# 選擇一些測試圖片進行預測
indices = np.random.choice(range(len(x_test)), size=5, replace=False)
test_images = x_test[indices]
true_labels = y_test[indices]

# 使用模型進行預測
model = load_model('cifar10_cnn_model.h5')
predictions = model.predict(test_images)

# 將結果可視化
plt.figure(figsize=(15, 6))
for i, (image, true_label, prediction) in enumerate(zip(test_images, true_labels, predictions)):
    plt.subplot(1, 5, i + 1)
    plt.imshow(image)
    plt.axis('off')
    true_class = class_labels[true_label[0]]
    predicted_class = class_labels[np.argmax(prediction)]
    plt.title(f"True: {true_class}\nPred: {predicted_class}", fontsize=10)
plt.tight_layout()
plt.show()
