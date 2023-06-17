import numpy as np
from  matplotlib import pyplot as plt
import time
from tensorflow.keras import  layers
from tensorflow.keras import  models
from tensorflow.keras import  optimizers
from tensorflow.keras import  datasets
from tensorflow.keras.layers import LeakyReLU


(x_train, y_train), (x_test, y_test) = datasets.cifar10.load_data()
assert x_train.shape == (50000, 32, 32, 3)
assert x_test.shape == (10000, 32, 32, 3)
assert y_train.shape == (50000, 1)
assert y_test.shape == (10000, 1)

epoch_num = 20

cnn = models.Sequential([
    #LeakyReLU(alpha= 0.01)
    layers.Conv2D(7, kernel_size=(3, 3), activation='relu', input_shape=(32, 32, 3), padding='same'),
    
    #layers.BatchNormalization(),
    layers.Conv2D(9, kernel_size=(3, 3), activation='relu'),

    layers.MaxPooling2D((2,2)),
    
    layers.Flatten(),
    layers.Dropout(0.3),
    layers.Dense(10, activation='softmax')
    ])

start_time = time.time()

cnn.summary()



#opt = optimizers.Adam(learning_rate = 0.1)
cnn.compile(optimizer='adam', loss= 'sparse_categorical_crossentropy',metrics=['accuracy'])
#hist = cnn.fit(x_train,y_train,epochs =epoch_num, validation_split=0.2, shuffle=True,batch_size= 128)
hist = cnn.fit(x_train,y_train,epochs =epoch_num, validation_split=0.2, shuffle=True)


print("--- %s seconds ---" % str((time.time() - start_time)/epoch_num))
print('Test',cnn.evaluate(x_test,y_test))



plt.plot(hist.history['loss'], linestyle = 'dotted',label = 'Train')
plt.plot(hist.history['val_loss'], linestyle = 'dotted',label = 'Validation')
plt.title('Loss')
plt.legend()
plt.show()

