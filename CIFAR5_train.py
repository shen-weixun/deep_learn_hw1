import tensorflow as tf
from tensorflow.keras import datasets, layers, models, regularizers
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from tensorflow.keras.utils import to_categorical
from sklearn.metrics import classification_report
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.metrics import classification_report, confusion_matrix
import numpy as np
num_classes=10
class_names = [
    'Airplane', 'Automobile', 'Bird', 'Cat', 'Deer',
    'Dog', 'Frog', 'Horse', 'Ship', 'Truck'
]

# 数据加载与预处理
(train_images, train_labels), (test_images, test_labels) = datasets.cifar10.load_data()
train_images, val_images, train_labels, val_labels = train_test_split(
    train_images, train_labels, test_size=0.2, random_state=42
)
train_images, val_images, test_images = train_images / 255.0, val_images / 255.0, test_images / 255.0
train_labels = to_categorical(train_labels, num_classes)
val_labels = to_categorical(val_labels, num_classes)
test_labels = to_categorical(test_labels, num_classes)
# 强化数据增强
train_datagen = tf.keras.preprocessing.image.ImageDataGenerator(
    rotation_range=15,
    width_shift_range=0.1,
    height_shift_range=0.1,
    horizontal_flip=True,
    zoom_range=0.1
)
train_generator = train_datagen.flow(train_images, train_labels, batch_size=256)

# 优化后的模型架构
model = models.Sequential([
    # 卷积模块1
    layers.Conv2D(64, (3,3), activation='relu', padding='same', input_shape=(32,32,3)),
    layers.Conv2D(64, (3,3), activation='relu', padding='same'),
    layers.BatchNormalization(),
    layers.MaxPooling2D((2,2)),
    layers.Dropout(0.3),
    
    # 卷积模块2
    layers.Conv2D(128, (3,3), activation='relu', padding='same'),
    layers.Conv2D(128, (3,3), activation='relu', padding='same'),
    layers.BatchNormalization(),
    layers.MaxPooling2D((2,2)),
    layers.Dropout(0.4),
    
    # 卷积模块3
    layers.Conv2D(256, (3,3), activation='relu', padding='same'),
    layers.BatchNormalization(),
    layers.MaxPooling2D((2,2)),
    layers.Dropout(0.5),
    
    # 输出层
    layers.Flatten(),
    layers.Dense(256, activation='relu'),
    layers.BatchNormalization(),
    layers.Dropout(0.5),
    layers.Dense(10, activation='softmax')
])
model.summary()

# 修改学习率并添加指数衰减
optimizer = tf.keras.optimizers.Adam(learning_rate=0.001, decay=1e-6)

# 修正损失函数参数（from_logits=False）
model.compile(optimizer=optimizer,
              loss=tf.keras.losses.categorical_crossentropy,
              metrics=['acc'])

# 添加回调函数
early_stop = tf.keras.callbacks.EarlyStopping(monitor='val_acc', patience=15, restore_best_weights=True)
reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=5, min_lr=1e-7)

# 训练模型（使用数据增强）
history = model.fit(
    train_generator,
    steps_per_epoch=len(train_images)//256,
    epochs=100,
    validation_data=(val_images, val_labels),
    callbacks=[early_stop, reduce_lr]
)

# 评估模型
test_loss, test_acc = model.evaluate(test_images, test_labels, verbose=0)
print(f"\nFinal Test Accuracy: {test_acc*100:.2f}%")
print(f"\nFinal Test loss: {test_loss*100:.2f}%")


y_pred = model.predict(test_images)
y_pred = np.argmax(y_pred, axis=1)
y_test = np.argmax(test_labels, axis=1)
cm = confusion_matrix(y_test, y_pred)


disp = ConfusionMatrixDisplay(confusion_matrix=cm,display_labels=class_names)

fig, ax = plt.subplots(figsize=(10, 10))
disp = disp.plot(xticks_rotation='vertical', ax=ax,cmap='Blues')
# 可视化训练过程
def plot_training(history):
    plt.figure(figsize=(12,4))
    plt.subplot(1,2,1)
    plt.plot(history.history['acc'], label='train')
    plt.plot(history.history['val_acc'], label='val')
    plt.title('Accuracy Curves')
    plt.legend()
    
    plt.subplot(1,2,2)
    plt.plot(history.history['loss'], label='train')
    plt.plot(history.history['val_loss'], label='val')
    plt.title('Loss Curves')
    plt.legend()
    plt.show()

plot_training(history)
