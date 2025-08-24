#%% 引用模組
import keras
import pandas as pd
import os
from PIL import Image
import numpy as np
from sklearn.model_selection import train_test_split
import tensorflow as tf
import matplotlib.pyplot as plt
'''
1.
CNN做圖像分類，在cnn layer後面加ann layer，最終得到圖分類各類別機率，如:是不是貓、狗?Unet是做圖像分割。最終是把圖切割成非常多的小格子，每個小格子的機率。可以做到區隔圖中多樣物件並標出物件不規則邊緣。

2.
模型長的像U形狀，結構分成左邊像下的Down，中間平的Bridge，右邊向上的UP，在Down和Up中間互相對應的Skip connection。Down和Cnn的Convorlution layer一樣提取出特徵圖並在Pooling layer進行壓縮。Down每一層中都包含兩個convorlution layer + 一個或兩個activation function + 一個pooling layer，第一層得出64張徵圖，傳給下一層得出128、256、512特徵圖。之所以每層為2 convolution + 1-2 activation + 1 pooling，不像Cnn每層1 convolution + 1 activation + 1 pooling。接續2層convolution原因為增加模型的深度及複雜性，可以提取更複雜的特徵，另外activation使用Relu : 使用非線性的激活函數。並且解決出現梯度消失問題。Down完傳至Bridge，通常為convorlution layer + activation function。Bridge作為Down encoder及Up decoder的連接。Down和Cnn的Convorlution layer一樣提取出特徵圖。Bridge從Down中所得的所有特徵圖，提取獲得更為細小、抽象的特徵。UP的每一層會透過Skip Connection與Down對應的每一層，一步步還原出把原圖上物件區隔開來及邊界標出的圖。Up每一層為一個Transposed Convolution(反卷層) + Skip Connection對應Down的一層 + 一個contracting path + 兩個convorlution layer + 一個或兩個activation function，在傳給下一層。Transposed Convolution在幫助放大特徵圖，並通過計算填補空缺像素格。從Down把特徵圖從Skip Connection傳過來，與Transposed Convolution傳上來的特徵圖在contracting path進行結合，原因為可以增加圖像特徵細節邊緣精準得辨識能力。後面接兩個convorlution layer + 一個或兩個activation function，在幫助從Transposed Convolution放大填補的像素格可能會比較粗糙，不夠精細的特徵，將重要的邊緣和細節信息進一步強化，提升最終輸出的質量。更好的幫助Skip Connection傳來的特徵圖及Transposed Convolution傳來得特徵圖做結合。這些過程中不斷減少特徵圖直到最後一層為一個convorlution layer做出每個小格子得機率分類。

3.
在上採樣中可以簡單使用UpSampling2D，通過重複象素放大圖像。Conv2DTranspose則會透過學習特徵進行放大填補空缺。strides決定放大倍數。Conv2DTranspose = UpSampling2D + Conv2D

4.
變形Unet :
    3DUnet : 作用於立體圖形的邊緣分割(MRI、CT)
    Attention Unet : 在原本的Unet中加入Attention gate
        把down的特徵圖及up的特徵圖引入，分別以比原本少的特徵圖數量進行學習(有助於捕獲最重要的特徵)，把分別學習過後的特徵圖進行結合，並且透過學習得出一張出有強調重點的特徵圖(注意力圖)。於輸出時原本Down特徵圖，每一張都會根據有強調重點的特徵圖去進行調整
    Nested U-Net

'''
random_state = 42
test_size = 0.1
target_size = (128, 128)

file = 'C:/研究所/自學/各模型/DATA/road/metadata.csv'
all_data = pd.read_csv(file)
data = all_data.loc[all_data.split == 'train']
image_folder = 'C:/研究所/自學/各模型/DATA/road/data'

x = []
y = []

for _, row in data.iterrows() :
    try :
        original_image_name = os.path.basename(row['sat_image_path'])
        segmented_image_name = os.path.basename(row['mask_path'])
        
        original_image_path = os.path.join(
            image_folder, original_image_name)
        segmented_image_path = os.path.join(
            image_folder, segmented_image_name)
    
        ori_image = Image.open(original_image_path)
        ori_image_resized = ori_image.resize(target_size, Image.BICUBIC)
        ori_image_resized_normalized_array = np.array(
            ori_image_resized, dtype = np.float32) / 255
        x.append(ori_image_resized_normalized_array)
        
        seg_image = Image.open(segmented_image_path)
        seg_image_resized = seg_image.resize(target_size, Image.NEAREST)
        seg_image_array = np.array(seg_image_resized)
        seg_image_binary = (seg_image_array == 0).astype(np.float32)
        if len(seg_image_binary.shape) == 3:
            seg_image_binary = np.all(seg_image_binary, axis=2).astype(np.float32)
        y.append(seg_image_binary)
        
    except Exception :
        continue

x_array = np.array(x)
y_array = np.array(y)

y_array = np.expand_dims(y_array, axis=-1)

x_train_val, x_test, y_train_val, y_test = train_test_split(
    x_array,
    y_array,
    random_state = random_state,
    test_size = test_size)

x_train, x_val, y_train, y_val = train_test_split(
    x_train_val,
    y_train_val,
    random_state = random_state,
    test_size = test_size)


train_data = tf.data.Dataset.from_tensor_slices((
    x_train, y_train))
train_data = train_data.cache().batch(
    batch_size = 32).prefetch(
        buffer_size = tf.data.AUTOTUNE)
val_data = tf.data.Dataset.from_tensor_slices((
    x_val, y_val))
val_data = val_data.cache().batch(
    batch_size = 32).prefetch(
        buffer_size = tf.data.AUTOTUNE)
        
def dice_loss(y_true, y_pred, smooth=  1e-6):
    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.cast(y_pred, tf.float32)
    intersection = tf.reduce_sum(y_true * y_pred)
    union = tf.reduce_sum(y_true) + tf.reduce_sum(y_pred)
    dice = (2. * intersection + smooth) / (union + smooth)
    return 1 - dice

def combined_loss(y_true, y_pred):
    return 0.5 * dice_loss(
        y_true, y_pred) + 0.5 * tf.keras.losses.binary_crossentropy(
            y_true, y_pred)
             
def conv_batch_block(inputs, num_filters) :
    x = keras.layers.Conv2D(
        num_filters, kernel_size = (3, 3), 
        activation = 'relu', padding = 'same')(inputs)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.Conv2D(
        num_filters, 3, activation='relu', padding='same')(x)
    x = keras.layers.BatchNormalization()(x)
    return x

def up_batch(inputs, num_filters) : 
    # z = keras.layers.UpSampling2D((2, 2))(inputs)
    # z = keras.layers.Conv2D(num_filters, kernel_size = (3, 3), activation = 'relu', padding = 'same')
    z = keras.layers.Conv2DTranspose(
        num_filters, kernel_size = (3, 3), 
        strides= (2, 2), padding = 'same')(inputs)
    z = keras.layers.BatchNormalization()(z)
    return z

def attention_gate(x, g, inter_channel):
    x = keras.layers.Conv2D(inter_channel, (1, 1), use_bias=False)(x)
    g = keras.layers.Conv2D(inter_channel, (1, 1), use_bias=False)(g)
    f = keras.layers.Activation('relu')(keras.layers.add([x, g]))
    f = keras.layers.Conv2D(1, (1, 1), activation = 'sigmoid', use_bias=False)(f)
    att = keras.layers.multiply([x, f])
    return att
    

def unet(input_size = x_train.shape[1 : ]):
    inputs = keras.layers.Input(shape = (input_size))
    # Encoder (Downsampling)
    c1 = conv_batch_block(inputs, 64)
    p1 = keras.layers.MaxPooling2D((2, 2))(c1)
    
    c2 = conv_batch_block(p1, 128)
    p2 = keras.layers.MaxPooling2D((2, 2))(c2)
    
    c3 = conv_batch_block(p2, 256)
    p3 = keras.layers.MaxPooling2D((2, 2))(c3)
    
    c4 = conv_batch_block(p3, 512)
    p4 = keras.layers.MaxPooling2D((2, 2))(c4)
    # Bridge
    c5 = conv_batch_block(p4, 1024)
    # Decoder (Upsampling)
    u6 = up_batch(c5, 512)
    a6 = attention_gate(c4, u6, 256)
    u6 = keras.layers.concatenate([u6, a6])
    c6 = conv_batch_block(u6, 512)
    
    u7 = up_batch(c6, 256)
    a7 = attention_gate(c3, u7, 128)
    u7 = keras.layers.concatenate([u7, a7])
    c7 = conv_batch_block(u7, 256)
    
    u8 = up_batch(c7, 128)
    a8 = attention_gate(c2, u8, 64)
    u8 = keras.layers.concatenate([u8, a8])
    c8 = conv_batch_block(u8, 128)
    
    u9 = up_batch(c8, 64)
    a9 = attention_gate(c1, u9, 32)
    u9 = keras.layers.concatenate([u9, a9])
    c9 = conv_batch_block(u9, 64)
    
    outputs = keras.layers.Conv2D(1, 1, activation='sigmoid')(c9)
    
    model = keras.models.Model(inputs=[inputs], outputs=[outputs])
    return model

model = unet()

model.compile(
    optimizer= keras.optimizers.Adam(learning_rate = 1e-4), 
    loss= combined_loss,
    metrics=[tf.keras.metrics.BinaryIoU(name='iou')],)

model.fit(
    train_data,
    epochs = 100, 
    validation_data = val_data,
    batch_size = 8)

model.summary()

y_train_pred = model.predict(x_train)
y_test_pred = model.predict(x_test)

train_biiou = keras.metrics.BinaryIoU(target_class_ids=[0, 1], threshold = 0.5)
train_biiou.update_state(y_train, y_train_pred)
train_biiou = train_biiou.result().numpy()

test_biiou = keras.metrics.BinaryIoU(target_class_ids=[0, 1], threshold = 0.5)
test_biiou.update_state(y_test, y_test_pred)
test_biiou = test_biiou.result().numpy()

print(f'train_iou : {train_biiou}')
print(f'test_iou : {test_biiou}')

# from tensorflow.keras import backend as K
# def dice_coef(y_true, y_pred):
#     y_true_f = K.flatten(y_true)
#     y_pred_f = K.flatten(y_pred)
#     intersection = K.sum(y_true_f * y_pred_f)
#     return (2 * intersection+1) / (K.sum(y_true_f) + K.sum(y_pred_f) + 1)

# train_dice_coef = dice_coef(y_train, y_train_pred)

index = 23
plt.figure(figsize=(15, 5))
# 原圖
plt.subplot(131)
plt.title('Original Image')
plt.imshow(x_test[index])
plt.axis('off')
# 真實分割圖
plt.subplot(132)
plt.title('True Mask')
plt.imshow(y_test[index].squeeze(), cmap='binary')
plt.axis('off')
# 預測的分割圖
plt.subplot(133)
plt.title('Predicted Mask')
y_test_pred_binary = (y_test_pred[index] > 0.5).squeeze().astype(int)
plt.imshow(y_test_pred_binary, cmap='binary')
plt.axis('off')
plt.tight_layout()
plt.show()



