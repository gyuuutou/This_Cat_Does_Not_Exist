import pickle
from PIL import Image
import numpy as np
import keras
import keras
from keras import Sequential
from keras.layers import *
from keras.optimizers import RMSprop
import os
import matplotlib.pyplot as plt
#モデルを保存するファイル名
generator_filename = "generator.h5"
discriminator_filename = "discriminator.h5"

def generator():
    if os.path.isfile(generator_filename):
        return keras.models.load_model(generator_filename)

    noise_shape = (100,)
    model = Sequential()

    #全結合 100→16*16*512
    model.add(Dense(512 * 16 * 16, activation="relu", input_shape=noise_shape))
    model.add(Reshape((16, 16, 512)))
    model.add(BatchNormalization(momentum=0.8))
    
    #転置畳み込み 16*16*512→32*32*256
    model.add(UpSampling2D())
    model.add(Conv2D(256, kernel_size=3, padding="same"))
    model.add(Activation("relu"))
    model.add(BatchNormalization(momentum=0.8))
    
    #転置畳み込み 32*32*256→64*64*128
    model.add(UpSampling2D())
    model.add(Conv2D(128, kernel_size=3, padding="same"))
    model.add(Activation("relu"))
    model.add(BatchNormalization(momentum=0.8))

    #転置畳み込み 64*64*128→64*64*3
    model.add(Conv2D(3, kernel_size=3, padding="same"))
    model.add(Activation("tanh"))

    model.summary()

    return model
    

def discriminator():
    if os.path.isfile(discriminator_filename):
        return keras.models.load_model(discriminator_filename)

    model = Sequential()

    #畳み込み 64*64*3->32*32*64
    model.add(Conv2D(64, kernel_size=3, strides=2, input_shape=(64,64,3), padding="same"))
    model.add(LeakyReLU(alpha=0.2))
    model.add(Dropout(0.25))

    #畳み込み 32*32*64->16*16*128
    model.add(Conv2D(128, (5,5), strides=2, padding="same"))
    model.add(LeakyReLU(alpha=0.2))
    model.add(Dropout(0.25))

    #畳み込み 16*16*128->8*8*256
    model.add(Conv2D(256, (5,5), strides=2, padding="same"))
    model.add(LeakyReLU(alpha=0.2))
    model.add(Dropout(0.25))

    #平坦化
    model.add(Flatten())

    #全結合
    model.add(Dense(1))
    model.add(Activation("sigmoid"))

    model.summary()

    return model

def dis_model(d):

    model = Sequential()

    model.add(d)

    model.compile(optimizer=RMSprop(lr=0.0002, decay=0.0), loss='binary_crossentropy', metrics=['accuracy'])

    return model



def adv_model(g,d):

    d.trainable = False
    model = Sequential([g, d])

    model.compile(optimizer=RMSprop(lr=0.0002, decay=0.0), loss='binary_crossentropy', metrics=['accuracy'])
    
    return model

#バッチ/データサイズはお使いのGPUとデータの大きさによって変えてください
#学習データは乱択で選択するので正確には指定エポック学習できるわけではありません
EPOCH = 1000
DATA_SIZE = 9993
BATCH = 256

if __name__ == "__main__":
    #pklファイルを格納しているパスに書き換えてください
    with open("./kaggle_cat/data.pkl", "rb") as f:
        dataset = pickle.load(f)

    fixed_noise = np.random.uniform(-1.0, 1.0, size=[25,100])

    #-1~1で正則化
    dataset = (dataset-127.5)/127.5

    g = generator()
    d = discriminator()

    dm = dis_model(d)
    am = adv_model(g,d)

    for i in range(int(DATA_SIZE/BATCH*EPOCH)):
        #識別器を学習させる
        noise = np.random.uniform(-1.0, 1.0, size=[BATCH,100])

        fake = g.predict(noise)

        true = dataset[np.random.randint(0, dataset.shape[0], BATCH)]

        train_img = np.concatenate([true, fake])
        train_label = np.concatenate([np.ones([BATCH,1]), np.zeros([BATCH,1])])

        d_loss = dm.train_on_batch(train_img, train_label)

        #生成器を学習させる
        train_noise = np.random.uniform(-1.0, 1.0, size=[BATCH,100])

        train_label = np.ones([BATCH])

        a_loss = am.train_on_batch(train_noise, train_label)
        
        #結果を表示
        log_mesg = "%d: [D loss: %f acc: %f] [A loss: %f acc: %f]" % (i+1, d_loss[0], d_loss[1], a_loss[0], a_loss[1])

        print(log_mesg)

        #500バッチごとに画像を生成してモデルを保存
        if i % 500 == 0:

                d.save("discriminator.h5")
                g.save("generator.h5")
                
                check_img = g.predict(fixed_noise)
                #生成された画像データを0-255の8bitのunsigned int型に収めます
                check_img =  check_img*127.5+127.5
                check_img = check_img.astype("uint8")

                # タイル状に num × num 枚配置
                num = 5
                # 空の入れ物（リスト）を準備
                d_list = []

                for j in range(len(check_img)):
                    img = Image.fromarray(check_img[j])
                    img = np.asarray(img)
                    d_list.append(img)
                # タイル状に画像を一覧表示
                fig, ax = plt.subplots(num, num, figsize=(10, 10))
                fig.subplots_adjust(hspace=0, wspace=0)
                for j in range(num):
                    for k in range(num):
                        ax[j, k].xaxis.set_major_locator(plt.NullLocator())
                        ax[j, k].yaxis.set_major_locator(plt.NullLocator())
                        ax[j, k].imshow(d_list[num*j+k], cmap="bone")
                #いい感じの保存名をつけましょう
                fig.savefig("~~~~~~~"+ str(i) + ".png")


