import cv2
import numpy as np
from operator import itemgetter


def display(img, frameName="OpenCV Image"):
    h, w = img.shape[0:2]
    neww = 800
    newh = int(neww*(h/w))
    img = cv2.resize(img, (neww, newh))
    cv2.imshow(frameName, img)
    cv2.waitKey(0)


def get_photos(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)
    _, th = cv2.threshold(gray, 210, 235, 1)
    cnts, _ = cv2.findContours(
        th.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    cnts = sorted(cnts, key=cv2.contourArea, reverse=True)
    area = img.shape[0]*img.shape[1]
    photos = []
    for i, c in enumerate(cnts):
        rect = cv2.minAreaRect(c)
        box = cv2.boxPoints(rect)
        box = np.array(box, dtype="int")
        if area/10 < cv2.contourArea(box) < area*2/3:
            width = int(rect[1][0])
            height = int(rect[1][1])
            cx, cy = np.mean(box, axis=0)
            src_pts = box.astype("float32")
            dst_pts = np.array([[0, height-1],
                                [0, 0],
                                [width-1, 0],
                                [width-1, height-1]], dtype="float32")
            M = cv2.getPerspectiveTransform(src_pts, dst_pts)
            photo = cv2.warpPerspective(img, M, (width, height))
            photos.append({
                'photo': photo,
                'x': cx,
                'y': cy,
            })
    photos = sorted(photos, key=itemgetter('x'))
    photos = sorted(photos, key=itemgetter('y'))
    photos = list(map(itemgetter('photo'), photos))
    return photos


if __name__ == '__main__':
    import os
    import argparse
    from tqdm import tqdm
    parser = argparse.ArgumentParser(
        prog='scanned-photos-extractor',
        description='Extracts photos from scanned image')
    parser.add_argument(
        'output_dir',
        help='directory in which photos will be stored')
    parser.add_argument(
        'files',
        nargs='*',
        help='scanned image files to be processed')
    args = parser.parse_args()
    newhead = args.output_dir
    for input_file in tqdm(args.files):
        head, tail = os.path.split(input_file)
        root, ext = os.path.splitext(tail)
        img = cv2.imread(input_file)
        photos = get_photos(img)
        nphotos = len(photos)
        for i, photo in enumerate(photos, 1):
            newtail = '{}_{}-{}{}'.format(root, i, nphotos, ext)
            output_file = os.path.join(newhead, newtail)
            ok = cv2.imwrite(output_file, photo)
            assert ok, "Could not write to file '{}'".format(output_file)
