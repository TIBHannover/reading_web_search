import cv2
import pandas as pd
from copy import copy


df_our = pd.read_csv("data/our_regressions.tsv", sep="\t")
df_cole2011 = pd.read_csv("data/cole2011_regressions.tsv", sep="\t")
df_word_level = pd.read_csv("data/words_map.tsv", sep="\t")

df_our = df_our[(df_our["uid"] == 48) & (df_our["url"].str.contains("weltderphysik"))]
df_cole2011 = df_cole2011[(df_cole2011["uid"] == 48) & (df_cole2011["url"].str.contains("weltderphysik"))]
df_word_level = df_word_level[(df_word_level["uid"] == 48) & (df_word_level["url"].str.contains("weltderphysik"))]

offset = -107

df_our["x"] += offset
df_our["mean_x"] += offset
df_cole2011["mean_x"] += offset
df_word_level["x"] += offset

image = cv2.imread("img/weltderphysik.png")

# draw bounding boxes for words
for idx, row in df_word_level.iterrows():
    rcolor = (80, 80, 80)
    thickness = 1
    try:
        top_left, bottom_right = (int(row["x"]), int(row["y"])), (int(row["x"] + row["width"]), int(row["y"] + row["height"]))
        image = cv2.rectangle(image, top_left, bottom_right, color=rcolor, thickness=thickness)
    except:
        ...

cv2.imwrite("results/qualitative_example_blank.png", image)


# paint our reading sequences and regressions
Img = copy(image)

indexe = df_our.index.values
i = 0

while i < df_our.shape[0]:
    cur = df_our.loc[indexe[i]].copy(deep=True)
    k = i+1
    try:
        nxt = df_our.loc[indexe[k]]
    except IndexError:
        break
    radius = 2 + int(0.5 * cur["duration"]/100)
    point = (int(cur["mean_x"] + radius/2), int(cur["mean_y"] + radius/2))
    point_thickness = 1
    line_thickness = 2

    if cur["regression"] & (cur["reading_sequence"] > -1):
        point_color = (255, 35, 40)
    elif ~(cur["regression"]) & (cur["reading_sequence"] > -1):
        point_color = (20, 222, 40)
    else:
        point_thickness = 1
        point_color = (160, 160, 160)

    if nxt["regression"] & (cur["reading_sequence"] > -1) & (nxt["reading_sequence"] > -1):
        line_color = (255, 35, 40)
    elif (cur["reading_sequence"] == nxt["reading_sequence"]) & (cur["reading_sequence"] > -1):
        line_color = (20, 222, 40)
    else:
        line_thickness = 1
        line_color = (160, 160, 160)

    try:
        point = (int(cur.x + cur.width/2), int(cur.y + cur.height/2))
        nxt_point = (int(nxt.x + nxt.width/2), int(nxt.y + nxt.height/2))
        draw_points_x = [point[0], int((point[0]+nxt_point[0])/2), nxt_point[0]]
        draw_points_y = [point[1], int((point[1]+nxt_point[1]+(point[1]-nxt_point[1]/5))/2), nxt_point[1]]

        if (cur["reading_sequence"] > -1) & (nxt["reading_sequence"] > -1):
            if nxt["regression"]:
                line_color = (255, 35, 40)
            elif cur["reading_sequence"] == nxt["reading_sequence"]:
                line_color = (20, 222, 40)
            else:
                raise Exception
            Img = cv2.line(Img, point, nxt_point, line_color, line_thickness)
    except:
        ...
    i = k

Img = cv2.addWeighted(Img, 0.7, image, 0.3, 0)
Img = Img[:,:,::-1]
cv2.imwrite("results/qualitative_example_our.png", Img)



# paint cole2011 reading sequences and regressions
Img = copy(image)

indexe = df_cole2011.index.values
i = 0

while i < df_cole2011.shape[0]:
    cur = df_cole2011.loc[indexe[i]].copy(deep=True)
    k = i+1
    try:
        nxt = df_cole2011.loc[indexe[k]]
    except IndexError:
        break
    radius = 2 + int(0.5 * cur["duration"]/100)
    point = (int(cur["mean_x"] + radius/2), int(cur["mean_y"] + radius/2))
    point_thickness = 1
    line_thickness = 2

    if cur["regression"] & (cur["reading_sequence"] > -1):
        point_color = (255, 35, 40)
    elif ~(cur["regression"]) & (cur["reading_sequence"] > -1):
        point_color = (20, 222, 40)
    else:
        point_thickness = 1
        point_color = (160, 160, 160)

    if nxt["regression"] & (cur["reading_sequence"] > -1) & (nxt["reading_sequence"] > -1):
        line_color = (255, 35, 40)
    elif (cur["reading_sequence"] == nxt["reading_sequence"]) & (cur["reading_sequence"] > -1):
        line_color = (20, 222, 40)
    else:
        line_thickness = 1
        line_color = (160, 160, 160)

    try:
        point = (int(cur.mean_x), int(cur.mean_y))
        nxt_point = (int(nxt.mean_x), int(nxt.mean_y))
        draw_points_x = [point[0], int((point[0]+nxt_point[0])/2), nxt_point[0]]
        draw_points_y = [point[1], int((point[1]+nxt_point[1]+(point[1]-nxt_point[1]/5))/2), nxt_point[1]]

        if (cur["reading_sequence"] > -1) & (nxt["reading_sequence"] > -1):
            if nxt["regression"]:
                line_color = (255, 35, 40)
            elif cur["reading_sequence"] == nxt["reading_sequence"]:
                line_color = (20, 222, 40)
            else:
                raise Exception
            Img = cv2.line(Img, point, nxt_point, line_color, line_thickness)
    except:
        ...
    i = k

# image = cv2.line(Img, start_point, end_point, color, thickness)
Img = cv2.addWeighted(Img, 0.7, image, 0.3, 0)
Img = Img[:,:,::-1]
cv2.imwrite("results/qualitative_example_cole2011.png", Img)

