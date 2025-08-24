#%% 引用模組
import numpy as np
import os
from sklearn.model_selection import train_test_split
import shutil
import pickle
import Augmentor
from sklearn import metrics
os.environ['KERAS_BACKEND'] = 'torch'
import keras
from keras.preprocessing.image import load_img, img_to_array
import keras_tuner
from sklearn.preprocessing import LabelEncoder
from sklearn.utils import shuffle
#%% 設定常見參數
random_state = 42
threshold = 0.5
test_size = 0.3
#%% 讀資料
data_dir = 'C:/研究所/自學/各模型/CNN圖檔/xray_dataset_covid19_prac/pic'
# 列出資料夾中資料夾名稱
class_dirs = os.listdir(data_dir)
# 把data_dir及class_dir結合成完整路徑，把分別路徑下'.jpg', '.png', '.jpeg'結尾的圖讀出來加到all_data
all_data = []
for class_dir in class_dirs:
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
    test_size = test_size, 
    random_state = random_state)

train_imgs, val_imgs = train_test_split(
    train_val_imgs, 
    test_size = test_size,
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
    class_temp_dir = os.path.join(temp_dir, class_dir)
    os.makedirs(
        class_temp_dir, 
        exist_ok = True)
    class_temp_dirs[class_dir] = class_temp_dir
# 複製訓練圖像到對應的臨時目錄
for img_path, class_label in train_data:
    dest_path = os.path.join(
        class_temp_dirs[class_label], os.path.basename(img_path))
    shutil.copy(img_path, dest_path)
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

normal_cnn = keras.Sequential([
    keras.layers.Input(
        shape = (X_train.shape[1:])),
    keras.layers.Conv2D(  
        filters = 32, kernel_size = (2, 2), activation = 'relu',
        strides = (1, 1), padding = 'same'),
    keras.layers.MaxPooling2D((3, 3)),
    keras.layers.BatchNormalization(),
    
    keras.layers.Conv2D(
        filters = 64, kernel_size = (2, 2), activation = 'relu',
        strides = (1, 1), padding = 'same'),
    keras.layers.MaxPooling2D((3, 3)),
    keras.layers.BatchNormalization(),
    
    keras.layers.Flatten(),
    keras.layers.Dense(64, activation = 'relu'),
    keras.layers.BatchNormalization(),
    keras.layers.Dense(32, activation = 'relu'),
    keras.layers.BatchNormalization(),
    keras.layers.Dense(1, activation = 'sigmoid')])

normal_cnn.compile(  
    optimizer =  keras.optimizers.Adam(learning_rate = 1e-2),
    loss = 'binary_crossentropy',
    metrics = ['accuracy'])

normal_cnn.fit(
    X_train,
    Y_train,
    epochs = 15,
    validation_data = (X_val, Y_val))

ytrain_cnn_proba = normal_cnn.predict(X_train)
ytest_cnn_proba = normal_cnn.predict(X_test)
# >0.5 : 1, <0.5 : 0
ytrain_cnn_predicted_labels = (ytrain_cnn_proba > threshold).astype(int)
ytest_cnn_predicted_labels = (ytest_cnn_proba  > threshold).astype(int)
train_accuracy = metrics.accuracy_score(
    Y_train, ytrain_cnn_predicted_labels)
test_accuracy = metrics.accuracy_score(
    Y_test, ytest_cnn_predicted_labels)
train_f1 = metrics.f1_score(
    Y_train, ytrain_cnn_predicted_labels)
test_f1 = metrics.f1_score(
    Y_test, ytest_cnn_predicted_labels)
train_auc = metrics.roc_auc_score(
    Y_train, ytrain_cnn_proba)
test_auc = metrics.roc_auc_score(
    Y_test, ytest_cnn_proba)
print(f' train_cnn Accuracy Score: {train_accuracy:.5f}')
print(f' test_cnn Accuracy Score: {test_accuracy:.5f}')
print(f' train_cnn F1 score: {train_f1:.5f}')
print(f' test_cnn F1 score: {test_f1:.5f}')
print(f' train_cnn AUC score: {train_auc:.5f}')
print(f' test_cnn AUC score: {test_auc:.5f}')
#%%
# 建模型
def tunning_cnn(hp):
    model = keras.Sequential()
    model.add(keras.layers.Input(
        shape = (X_train.shape[1 : ])))
    nums_cnn_layers = hp.Int('nums_cnn_layers', 1, 4, 1)
    start_filter = hp.Int('nums_start_filter', 8, 16, 2)
    multiple_filters = hp.Float('multiple_filters', 1, 2, 0.25)
    for i in range(nums_cnn_layers):
        current_filters = int(start_filter * multiple_filters ** i )
        model.add(keras.layers.Conv2D(
            filters = current_filters,
            kernel_size = hp.Choice('start_kernel_size', 
                                    values = [2, 3]),
            strides = (1, 1),
            padding = 'same',
            activation = hp.Choice('start_activation',
                                   values = ['relu', 'tanh'])))
        model.add(keras.layers.MaxPooling2D(
            pool_size = hp.Choice('start_pool_size',
                                  values = [2, 3])))
        model.add(keras.layers.BatchNormalization())
                  
    nums_ann_layers = hp.Int('nums_ann_layers', 1, 3, step = 1)
    model.add(keras.layers.Flatten())        
    for i in range(nums_ann_layers):
        model.add(keras.layers.Dense(
            units = hp.Int(f' units_{i}', 2, 100, step = 4),
            activation = hp.Choice(f' activation_{i}', 
                                   values = ['relu', 'tanh'])))
        model.add(keras.layers.BatchNormalization())
            
    model.add(keras.layers.Dense(1, activation = 'sigmoid'))
    choiced_optimizer = hp.Choice('optimizer', 
                                  values = [
                                      'adam', 'sgd', 'rmsprop', 'adamw'])
    choiced_learning_rate = hp.Float(
        'learning_rate', 
        1e-4, 1e-3, 
        sampling = 'log')
        
    if choiced_optimizer == 'adam':
        optimizer = keras.optimizers.Adam(
            learning_rate = choiced_learning_rate)
    elif choiced_optimizer == 'sgd':
        optimizer = keras.optimizers.SGD(
            learning_rate = choiced_learning_rate)
    elif choiced_optimizer == 'rmsprop':
        optimizer = keras.optimizers.RMSprop(
            learning_rate = choiced_learning_rate)
    elif choiced_optimizer == 'adamw':
        optimizer = keras.optimizers.AdamW(
            learning_rate = choiced_learning_rate)
            
    model.compile(  
        optimizer = optimizer,
        loss = 'binary_crossentropy',
        metrics = [
            keras.metrics.BinaryAccuracy(
                name = 'accuracy')])
    return model
early_stopping = keras.callbacks.EarlyStopping(
    monitor = 'val_accuracy',
    mode = 'max',
    patience = 20,
    restore_best_weights = True)
# 創建BayesianOptimization
bayes_tuning_cnn = keras_tuner.BayesianOptimization(
    tunning_cnn,
    objective = 'val_accuracy',
    max_trials = 50,
    num_initial_points = 10)
# 把資料FIT進去找
bayes_tuning_cnn.search(
    X_train,
    Y_train,
    epochs = 150,
    validation_data = (X_val, Y_val),
    callbacks = [early_stopping])
# 最佳模型
best_bayes_tuning_cnn_model = bayes_tuning_cnn.get_best_models(
    num_models = 1)[0]
# 最佳超參數
best_bayes_tuning_cnn_hp = bayes_tuning_cnn.get_best_hyperparameters(
    num_trials = 1)[0]
# 存檔
file_name = "C:/Users/88691/Desktop/自學/AI/practice/best_bayes_tuning_cnn_model.pkl"
pickle.dump(best_bayes_tuning_cnn_model, open(file_name, "wb"))
best_bayes_tuning_cnn_model = pickle.load(open(file_name, "rb"))
file_name = "C:/Users/88691/Desktop/自學/AI/practice/best_bayes_tuning_cnn_hp.pkl"
pickle.dump(best_bayes_tuning_cnn_hp, open(file_name, "wb"))
best_bayes_tuning_cnn_hp = pickle.load(open(file_name, "rb"))

ytrain_best_tuning_cnn_proba = best_bayes_tuning_cnn_model.predict(X_train)
ytest_best_tuning_cnn_proba = best_bayes_tuning_cnn_model.predict(X_test)
# >0.5 : 1, <0.5 : 0
ytrain_tuning_cnn_predicted_labels = (ytrain_cnn_proba > threshold).astype(int)
ytest_tuning_cnn_predicted_labels = (ytest_cnn_proba > threshold).astype(int)
train_accuracy = metrics.accuracy_score(
    Y_train, ytrain_tuning_cnn_predicted_labels)
test_accuracy = metrics.accuracy_score(
    Y_test, ytest_tuning_cnn_predicted_labels)
train_f1 = metrics.f1_score(
    Y_train, ytrain_tuning_cnn_predicted_labels)
test_f1 = metrics.f1_score(
    Y_test, ytest_tuning_cnn_predicted_labels)
train_auc = metrics.roc_auc_score(
    Y_train, ytrain_cnn_proba)
test_auc = metrics.roc_auc_score(
    Y_test, ytest_cnn_proba)
print(f' train_tuning_cnn Accuracy Score: {train_accuracy:.5f}')
print(f' test_tuning_cnn Accuracy Score: {test_accuracy:.5f}')
print(f' train_tuning_cnn F1 score: {train_f1:.5f}')
print(f' test_tuning_cnn F1 score: {test_f1:.5f}')
print(f' train_tuning_cnn AUC score: {train_auc:.5f}')
print(f' test_tuning_cnn AUC score: {test_auc:.5f}')