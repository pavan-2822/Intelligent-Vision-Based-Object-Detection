from lxml import etree
import os
import cv2

def generate_annotation(save_dir, base_name, frame, detections, label_map):
    formatted_boxes = []
    for det in detections:
        formatted_boxes.append([
            int(det[0]), int(det[1]), int(det[2]), int(det[3]), label_map[int(det[5])]
        ])

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    os.chdir(save_dir)

    output_img = "XML_" + base_name + ".png"
    cv2.imwrite(output_img, frame)

    root = etree.Element("annotation")

    folder_tag = etree.Element("folder")
    folder_tag.text = os.path.basename(os.getcwd())
    root.append(folder_tag)

    file_tag = etree.Element("filename")
    file_tag.text = output_img
    root.append(file_tag)

    path_tag = etree.Element("path")
    path_tag.text = os.path.join(os.getcwd(), output_img.split(".")[0] + ".jpg")
    root.append(path_tag)

    source_tag = etree.Element("source")
    db_tag = etree.Element("database")
    db_tag.text = "Unknown"
    source_tag.append(db_tag)
    root.append(source_tag)

    size_tag = etree.Element("size")
    image = cv2.imread(output_img)
    
    width_tag = etree.Element("width")
    width_tag.text = str(image.shape[1])
    
    height_tag = etree.Element("height")
    height_tag.text = str(image.shape[0])
    
    depth_tag = etree.Element("depth")
    depth_tag.text = str(image.shape[2])

    size_tag.extend([width_tag, height_tag, depth_tag])
    root.append(size_tag)

    segmented_tag = etree.Element("segmented")
    segmented_tag.text = "0"
    root.append(segmented_tag)

    for entry in formatted_boxes:
        label = entry[4]
        xmin, ymin, xmax, ymax = map(lambda v: str(int(float(v))), entry[:4])

        obj_tag = etree.Element("object")

        name_tag = etree.Element("name")
        name_tag.text = label
        obj_tag.append(name_tag)

        pose_tag = etree.Element("pose")
        pose_tag.text = "Unspecified"
        obj_tag.append(pose_tag)

        truncated_tag = etree.Element("truncated")
        truncated_tag.text = "0"
        obj_tag.append(truncated_tag)

        difficult_tag = etree.Element("difficult")
        difficult_tag.text = "0"
        obj_tag.append(difficult_tag)

        box_tag = etree.Element("bndbox")
        for coord_name, val in zip(["xmin", "ymin", "xmax", "ymax"], [xmin, ymin, xmax, ymax]):
            tag = etree.Element(coord_name)
            tag.text = val
            box_tag.append(tag)

        obj_tag.append(box_tag)
        root.append(obj_tag)

    xml_bytes = etree.tostring(root, pretty_print=True)
    with open(output_img.split(".")[0] + ".xml", "wb") as xml_file:
        xml_file.write(xml_bytes)

    os.chdir("..")
#