import cv2
import numpy as np

# 8ビット1チャンネルのグレースケールとして画像を読み込む
img = cv2.imread("sample1.jpg", cv2.IMREAD_GRAYSCALE)[10:-10, 10:-10]
ret, img = cv2.threshold(img, 150, 255, cv2.THRESH_BINARY)

contours, hierarchy = cv2.findContours(
    img, 
    cv2.RETR_LIST,      # 一番外側の輪郭のみを取得する 
    cv2.CHAIN_APPROX_NONE   # 輪郭座標の省略なし
    ) 

# 画像表示用に入力画像をカラーデータに変換する
img_disp = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

# 輪郭の点の描画
for i, contour in enumerate(contours):
    # 輪郭を描画
    cv2.drawContours(img_disp, contours, i, (255, 0, 0), 2)

    # 傾いていない外接する矩形領域
    x,y,w,h = cv2.boundingRect(contour)
    cv2.rectangle(img_disp,(x,y),(x+w-1,y+h-1),(0,255,0),2)

    # 傾いた外接する矩形領域
    rect = cv2.minAreaRect(contour)
    box = cv2.boxPoints(rect)
    print(rect, box)
    box = np.intp(box)
    cv2.drawContours(img_disp,[box],0,(0,0,255), 2)
    
# 画像の表示
cv2.imshow("Image", img_disp)

# キー入力待ち(ここで画像が表示される)
cv2.waitKey()