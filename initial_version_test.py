# 测试一下，识别当前目录下的test.jpg文件
import matplotlib.pyplot as plt 
from keras.preprocessing import image 
import os 
from keras.models import load_model
from keras.applications import VGG16 

conv_base = VGG16(weights='imagenet', include_top=False, input_shape=(150, 150, 3)) 

def predict(image_path):   # 测试所用.jpg图片文件所在的文件夹
    # 加载全连接层模型（为毛不能用相对路径了？）
    model = load_model(r'C:\Users\sanzh\Desktop\深度学习资料\pythonDL\convnet\trash_classification\output\initial_version.h5')

    
    files = os.listdir(image_path)
    trash_dict = {0: '其他垃圾', 1: '厨余垃圾', 2: '可回收垃圾', 3: '有害垃圾'}

    for i in range(len(files)): 
        original_image = image.load_img(image_path + files[i])
        img = image.load_img(image_path + files[i], target_size=(150, 150))
        x = image.img_to_array(img) 
        x = x.astype('float32') / 255.0 
        x = x.reshape((1, ) + x.shape) 
        res = conv_base.predict(x)  # 提取VGG16对这张图给出的特征
        res = res.reshape(1, 8192)
        res = model.predict(res)    # 以特征为全连接层的输入
        res = res.argmax()          # 选出4种垃圾中概率最大的那一个
        
        text = trash_dict[res]
        plt.subplot(int(len(files)/4)+1, 4, i+1)
        plt.axis("off")
        plt.imshow(original_image)
        plt.text(30, 0, text, fontdict=dict(fontsize=15, color='r',
                    family='FangSong',#字体,可选'serif', 'sans-serif', 'cursive', 'fantasy', 'monospace'
                    weight='heavy',#磅值，可选'light', 'normal', 'medium', 'semibold', 'bold', 'heavy', 'black'
                    )#字体属性设置
        )
    plt.show() 

# 存放jpg图片的文件夹
dir_path = 'C:\\Users\\sanzh\\Desktop\\深度学习资料\\pythonDL\\convnet\\trash_classification\\test\\trash_img\\'
predict(dir_path)
