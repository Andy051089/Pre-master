from tensorflow.keras.preprocessing.image import ImageDataGenerator

def create_aug_gen(X, Y, batch_size=32):
    data_gen_args = dict(rotation_range=10,
                         width_shift_range=0.1,
                         height_shift_range=0.1,
                         zoom_range=0.2,
                         horizontal_flip=True,
                         fill_mode='nearest')
    image_datagen = ImageDataGenerator(**data_gen_args)
    mask_datagen = ImageDataGenerator(**data_gen_args)
       
    image_generator = image_datagen.flow(X, batch_size=batch_size, seed=42)
    mask_generator = mask_datagen.flow(Y, batch_size=batch_size, seed=42)
       
    return zip(image_generator, mask_generator)


aug_gen = create_aug_gen(x_train, y_train, batch_size=8)
model.fit(aug_gen, ...)
   
