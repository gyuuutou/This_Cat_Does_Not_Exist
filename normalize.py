import glob
from PIL import Image
import os
import math

for f in glob.glob("./archive/*/*.jpg"):

    #画像をPILで読み込む
    img = Image.open(f)

    #猫の顔パーツの位置を読み込む
    fp = open( f + ".cat", "r")
    posi = [int(x) for x in fp.read().strip().split()]
    fp.close()

    #目が水平に来るように回転
    
    mid_x = (posi[1]+posi[3])/2 #回転の中心x
    mid_y = (posi[2]+posi[4])/2 #回転の中心y
    
    angle = math.degrees( math.atan2((posi[4]-posi[2]),(posi[3]-posi[1])) ) #回転させる角度

    img = img.rotate(angle,center=(mid_x,mid_y))


    #目と同じ位置に来るように拡大

    length = math.sqrt((posi[4]-posi[2])**2 + (posi[3]-posi[1])**2)

    ratio = 32/length

    img = img.resize((int(img.width*ratio), int(img.height*ratio)))

    #トリミング

    mid_x*=ratio
    mid_y*=ratio

    img = img.crop((mid_x-32, mid_y-32, mid_x+32, mid_y+32))

    #指定したディレクトリに保存
    file_name = os.path.basename(str(f))
    img.save("./torch_data/data/" + file_name)
    

    
