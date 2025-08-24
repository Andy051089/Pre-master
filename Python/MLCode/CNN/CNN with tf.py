#%% 引入模組
import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split
import tensorflow as tf
tf.config.set_visible_devices([], 'GPU')
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import shutil
import keras
from sklearn import metrics
import Augmentor
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from sklearn.preprocessing import LabelEncoder
from sklearn.utils import shuffle
#%% CNN
'''
1.CNN主要是做圖片的預測分類辨識。如果把圖片放大後，可以是一小格一小格的像素點，每格包含R、G、B的3組數字組成。當在訓練CNN模型時，
主要分為Convolution Layer和Pooling Layer。Convolution Layer中模型會通過計算，把整張圖辨識拆成各項特徵，擷取出圖中各項某
些部分可以辨識的特徵。並且為了泛化會把各項特徵周圍一起進行學習訓練。Convolution Layer後接著Pooling Layer，主要是把上一層
Convolution Layer擷取的各項特徵，進行降低緯度，讓圖變模糊及縮小，更方便計算及泛化未來資料。再傳入下一個Convolution Layer，
把各項特徵結合在一起，組合成各項特徵的結合，最終把各項特徵壓扁結合，作為ANN的訓練資料。在特徵偵測擷取時，以幾*幾的方格作為每個特
徵的大小，以strides設定的移動格數把圖的每個地方都至少輪過一次，但較靠近圖中央會被方格反覆偵測到，而周圍次數少，padding是在原圖
的周圍加上空白的格子，讓周圍的圖可以增加偵測到的次數。做不做Pooling在後面特徵擷取計算時都會得到相同大小。為了增加圖像的預測能力
，訓練資料的圖片是可以平移、旋轉、縮放、亮度調整等。

2.一次epochs，會從train_generator拿設定的batch_size張圖，進行訓練後再繼續拿設定的batch_size張圖，直到張數達原本資料及中圖
片的數目。所拿的圖不一定是原本資料夾的圖。訓練完後進入驗證過程，從test_generator拿設定的batch_size張圖，直到資料夾中的張數，
再進入到下一次epochs。

3.原圖大小假如是28*28，filters = 64, kernel_size = (5, 5)，那就會使用64張5*5的過濾器，對原圖擷取特徵產生64張28*28的特徵圖傳給POOLING LAYER，POOLING LAYER把全部64張28*28約縮小成設定Pooling(3, 3)3倍28/3約等9，大約成64張9*9的特徵圖
kernel_size : 複雜任務傾向於使用3x3(捕捉更複雜的特徵)但容易Over fitting，簡單任務可以考慮2x2

4.
rescale : 把所有ImageDataGenerator讀進來的圖都做標準化/255
.flow_from_directory : 為數據生成器
target_size : 圖像尺寸，大多模型是讀224,244
class_mode : binary為二分類，categorical為多分類
shuffle : 是否打散資料(為了防止訓練資料有順序性)
filter : 要幾個特徵 
kernel_size : 每個特徵的大小(2*2、3*3)
rotation_range : 轉多少角度
width_shift_range : 左右移動
height_shift_range : 上下移動
shear_range : 縮小
zoom_range : 放大
horizontal_flip : 垂直翻轉
vertical_flip : 左右翻轉
brightness_range : 亮度調整
fill_mode : nearest在做圖片翻轉平移時如果出現空白像素，使用附近填空
steps_per_epoch = len(train_generator) : 每個epoch中會把訓練圖全部看一次
validation_steps = len(val_generator) : 在驗證時會把訓練圖全部看一次
其餘可見ANN
'''
#%% 設定常見參數
random_state = 42
threshold = 0.5
test_size = 0.3
#%% For Tensorflow
# 原始文件地方
data_dir = 'C:/研究所/自學/各模型/CNN圖檔/xray_dataset_covid19_prac/pic'
# 列出資料夾中資料夾名稱
class_dirs = os.listdir(data_dir)
# 把data_dir及class_dir結合成完整路徑，把分別路徑下'.jpg', '.png', '.jpeg'結尾的圖讀出來加到all_data
all_data = []
for class_dir in os.listdir(data_dir):
    class_path = os.path.join(
        data_dir, class_dir)
    if os.path.isdir(class_path):
        images = [img for img in os.listdir(class_path) if img.endswith(
            ('.jpg', '.png', '.jpeg'))]
        all_data.extend(
            [(os.path.join(
                class_path, img), class_dir) for img in images])
# 把資料切割成訓練驗證測試
train_val_imgs, test_imgs = train_test_split(
    all_data, 
    test_size = 0.2, 
    random_state = random_state)

train_imgs, val_imgs = train_test_split(
    train_val_imgs, 
    test_size = 0.1,
    random_state = random_state)

train_data = []
test_data = []
val_data = []
augmented_train_data = []
# 把分出的資料加到list
train_data.extend(train_imgs)
test_data.extend(test_imgs)
val_data.extend(val_imgs)
# 創建臨時目錄來存儲每個類別的增強訓練圖像
temp_dir = "temp_augmentation_dir"
os.makedirs(
    temp_dir, exist_ok = True)
# 為每個類別創建子目錄
class_temp_dirs = {}
for class_dir in class_dirs:
    class_temp_dir = os.path.join(
        temp_dir, class_dir)
    os.makedirs(
        class_temp_dir, exist_ok = True)
    class_temp_dirs[class_dir] = class_temp_dir
# 複製訓練圖像到對應的臨時目錄
for img_path, class_label in train_data:
    dest_path = os.path.join(
        class_temp_dirs[class_label], os.path.basename(img_path))
    shutil.copy(
        img_path, dest_path)
# 為每個類別創建一個Pipeline並進行數據增強
for class_label, class_temp_dir in class_temp_dirs.items():
    pipeline = Augmentor.Pipeline(class_temp_dir)
# 設置增強操作
    pipeline.rotate(
        probability = 0.7, 
        max_left_rotation = 10, 
        max_right_rotation = 10)
    # pipeline.flip_left_right(probability = 0.5)
    # pipeline.flip_top_bottom(probability = 0.5)
    pipeline.flip_left_right(probability = 0.5)
    pipeline.zoom_random(
        probability = 0.5, 
        percentage_area = 0.8)
    pipeline.random_brightness(
        probability = 0.5, 
        min_factor = 0.7, 
        max_factor = 1.3)
    pipeline.random_contrast(
        probability = 0.5, 
        min_factor = 0.7, 
        max_factor = 1.3)
# 每個類別生成幾張
    pipeline.sample(300)
# 獲取增強後的圖片路徑
    augmented_dir = os.path.join(
        class_temp_dir, "output")
    for img_name in os.listdir(augmented_dir):
        img_path = os.path.join(
            augmented_dir, img_name)
        augmented_train_data.append(
            (img_path, class_label))
# 合併原始訓練數據和增強後的數據
train_data.extend(augmented_train_data)
#%%
# 建立資料的生成器並初始化功能
train_datagen = ImageDataGenerator(rescale = 1. / 255)
# 建立for驗證資料的生成器並初始化功能
val_datagen = ImageDataGenerator(rescale = 1. / 255)
# 建立for測試資料的生成器並初始化功能
test_datagen = ImageDataGenerator(rescale = 1./255)
# 建立訓練生成器
train_generator = train_datagen.flow_from_dataframe(
    dataframe = pd.DataFrame(
        train_data, columns = ['filename', 'class']),
    x_col = 'filename',
    y_col = 'class',
    target_size = (64, 64),
    batch_size = 64,
    class_mode = 'binary',
    shuffle = True)
# 建立測試生成器
test_generator = test_datagen.flow_from_dataframe(
    dataframe = pd.DataFrame(
        test_data, columns = ['filename', 'class']),
    x_col = 'filename',
    y_col = 'class',
    target_size = (64, 64),
    batch_size = 32,
    class_mode = 'binary',
    shuffle = False)
# 建立驗證生成器
val_generator = val_datagen.flow_from_dataframe(
    dataframe = pd.DataFrame(
        val_data, columns = ['filename', 'class']),
    x_col = 'filename',
    y_col = 'class',
    target_size = (64, 64),
    batch_size = 7,
    class_mode = 'binary',
    shuffle = False)
#%%
target_size = (224, 224)
X_train = []
Y_train = []
for img_path, label in train_data:
    img = load_img(img_path, target_size = target_size)
    img_array = img_to_array(img)
    X_train.append(img_array)
    Y_train.append(label)
X_train = np.array(X_train)
Y_train = np.array(Y_train)

X_test = []
Y_test = []
for img_path, label in test_data:
    img = load_img(img_path, target_size = target_size)
    img_array = img_to_array(img)
    X_test.append(img_array)
    Y_test.append(label)
X_test = np.array(X_test)
Y_test = np.array(Y_test)

# 處理驗證數據
X_val = []
Y_val = []
for img_path, label in val_data:
    img = load_img(img_path, target_size = target_size)
    img_array = img_to_array(img)
    X_val.append(img_array)
    Y_val.append(label)
X_val = np.array(X_val)
Y_val = np.array(Y_val)

label_encoder = LabelEncoder()
Y_train = label_encoder.fit_transform(Y_train)
Y_val = label_encoder.fit_transform(Y_val)
Y_test = label_encoder.fit_transform(Y_test)

X_train = X_train / 255
X_test = X_test / 255
X_val = X_val / 255
# 清理臨時目錄
shutil.rmtree(temp_dir)
keras.utils.set_random_seed(random_state)

X_train, Y_train = shuffle(
    X_train, Y_train, 
    random_state = random_state)
#%% 建立一般CNN
import time
start = time.time()
normal_cnn = keras.Sequential([
    keras.layers.Input(shape = (X_train.shape[1 : ])),
    keras.layers.Conv2D(  
        filters=16, kernel_size=(3, 3), activation='relu',
        strides=(1, 1), padding='same'),
    keras.layers.BatchNormalization(),
    keras.layers.MaxPooling2D((3, 3)),
    
    keras.layers.Conv2D(
        filters=32, kernel_size=(3, 3), activation='relu',
        strides=(1, 1), padding='same'),
    keras.layers.BatchNormalization(),
    keras.layers.MaxPooling2D((3, 3)),
    
    keras.layers.Conv2D(
        filters=64, kernel_size=(3, 3), activation='relu',
        strides=(1, 1), padding='same'),
    keras.layers.BatchNormalization(),
    keras.layers.MaxPooling2D((2, 2)),
    
    keras.layers.Conv2D(
        filters=128, kernel_size=(3, 3), activation='relu',
        strides=(1, 1), padding='same'),
    keras.layers.BatchNormalization(),
    keras.layers.MaxPooling2D((2, 2)),
    
    keras.layers.Flatten(),
    keras.layers.Dense(32, activation='relu'),
    keras.layers.BatchNormalization(),
    keras.layers.Dense(16, activation='relu'),
    keras.layers.BatchNormalization(),
    keras.layers.Dense(1, activation='sigmoid')])

normal_cnn.compile(  
    optimizer =  keras.optimizers.Adam(learning_rate = 1e-2),
    loss = 'binary_crossentropy',
    metrics = ['accuracy'])

normal_cnn.fit(
    X_train,
    Y_train,
    epochs = 50, 
    validation_data = ((X_val, Y_val)))
# normal_cnn.evaluate(test_generator)
end = time.time()
end-start

# 清理臨時目錄
shutil.rmtree(temp_dir)

cnn_train_images, cnn_train_labels = [], []
for i in range(len(train_generator)):
    x_batch, y_batch = train_generator[i]
    cnn_train_images.extend(x_batch)
    cnn_train_labels.extend(y_batch)
cnn_train_images = np.array(cnn_train_images)
cnn_train_labels = np.array(cnn_train_labels)

cnn_test_images, cnn_test_labels = [], []
for i in range(len(test_generator)):
    x_batch, y_batch = test_generator[i]
    cnn_test_images.extend(x_batch)
    cnn_test_labels.extend(y_batch)
cnn_test_images = np.array(cnn_test_images)
cnn_test_labels = np.array(cnn_test_labels)

ytrain_cnn_proba = normal_cnn.predict(cnn_train_images)
ytest_cnn_proba = normal_cnn.predict(cnn_test_images)
# >0.5 : 1, <0.5 : 0
ytrain_cnn_predicted_labels = (ytrain_cnn_proba > threshold).astype(int)
ytest_cnn_predicted_labels = (ytest_cnn_proba > threshold).astype(int)
train_accuracy = metrics.accuracy_score(
    cnn_train_labels, ytrain_cnn_predicted_labels)
test_accuracy = metrics.accuracy_score(
    cnn_test_labels, ytest_cnn_predicted_labels)
train_f1 = metrics.f1_score(
    cnn_train_labels, ytrain_cnn_predicted_labels)
test_f1 = metrics.f1_score(
    cnn_test_labels, ytest_cnn_predicted_labels)
train_auc = metrics.roc_auc_score(
    cnn_train_labels, ytrain_cnn_proba)
test_auc = metrics.roc_auc_score(
    cnn_test_labels, ytest_cnn_proba)
print(f' train_cnn Accuracy Score: {train_accuracy:.5f}')
print(f' test_cnn Accuracy Score: {test_accuracy:.5f}')
print(f' train_cnn F1 score: {train_f1:.5f}')
print(f' test_cnn F1 score: {test_f1:.5f}')
print(f' train_cnn AUC score: {train_auc:.5f}')
print(f' test_cnn AUC score: {test_auc:.5f}')
# 清理臨時目錄
shutil.rmtree(temp_dir)