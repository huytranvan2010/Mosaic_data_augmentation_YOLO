import cv2
import os

def preprocess(image_name, image_dir, label_dir):
    """
        Get image path, label path, label name
        image_name: For exxmaple img_0.png
        image_dir: original image directory
        label_dir: orrigianl label directory
    """
    image_path = os.path.sep.join([image_dir, image_name])
    # get name of image
    name = image_name.split('.')[0]
    label_name = name + '.txt'
    label_path = os.path.sep.join([label_dir, label_name])

    return image_path, label_path, label_name

def draw_rect(img, bboxes, color=(255, 0, 0)):
    """
        Draw bounding boxes on an image
    """
    # copy image
    img = img.copy()
    height, width = img.shape[:2]
    for bbox in bboxes:
        center_x, center_y, w, h = bbox[0], bbox[1], bbox[2], bbox[3]
        x = int((center_x - w/2) * width)
        w = int(w * width)
        y = int((center_y - h/2) * height)
        h = int(h * height)
        img = cv2.rectangle(img, (x, y), (x+w, y+h), color, 1)
    return img

def read_img(image_path, cvt_color=True):
    """
        Read image, convert from BGR (OpenCV) to RGB (Albumentation use that)
    """
    img = cv2.imread(image_path)
    if cvt_color:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # thấy ví dụ cần chuyển
    return img

def save_img(image, save_path, jpg_quality=None):
    """
        Save image with option to compress image
    """
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)   # do ảnh từ albumentation là RGB

    if jpg_quality:
        cv2.imwrite(save_path, image, [int(cv2.IMWRITE_JPEG_QUALITY), jpg_quality])
    else:
        cv2.imwrite(save_path, image)

def display_img(image_path, label_path):
    """
        Display image with bounding boxes
    """
    img = read_img(image_path, cvt_color=False)
    bboxes = read_label(label_path)[0]
    img = draw_rect(img, bboxes)
    cv2.imshow('Image', img)
    cv2.waitKey(0)

def read_label(label_path):
    """
        Read label from label file .txt (YOLO format)
    """
    with open(label_path) as f:
        conts = f.readlines()

    bboxes = []
    class_labels = []
    for cont in conts:
        cont = cont.strip().split()     # convert to list, for example ['0', '0.273047', '0.490972', '0.038281', '0.087500']
        # yolo format
        center_x, center_y, w, h = float(cont[1]), float(cont[2]), float(cont[3]), float(cont[4])
        bboxes.append([center_x, center_y, w, h])
        class_labels.append(cont[0])
    return (bboxes, class_labels)

def save_label(bboxes, class_labels, label_path):
    """
        Save the label
    """
    # create an empty list
    tem_lst = []
    for i, bbox in enumerate(bboxes):
        tem_lst.append(class_labels[i] + ' ' + str(bbox[0]) + ' ' + str(bbox[1]) + ' ' + str(bbox[2]) + ' ' + str(bbox[3]) + '\n')

    # write to the file
    with open(label_path, 'w') as f:
        f.writelines(tem_lst)

"""
    Chuyển bboxes ở dạng list (nested list) và class_labels ở dạng list để tạo lại file label
    ['1 0.221094 0.339583 0.051562 0.101389\n', '1 0.530469 0.517361 0.053125 0.129167\n']
    Khi đọc hết file label sẽ có dạng như này, tận dụng nó để tạo lại file label
"""