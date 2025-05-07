import os
from tqdm import tqdm
from lxml import etree
import cv2

base_dir = os.path.basename(os.getcwd())
if base_dir == "tools":
    os.chdir("..")

root_dataset = "OIDv4_ToolKit/OID/Dataset"

def run_conversion():
    project_root = os.getcwd()
    os.chdir(root_dataset)
    categories = os.listdir(os.getcwd())

    for category in categories:
        if os.path.isdir(category):
            os.chdir(category)
            print("Processing Category:", category)
            subdirs = os.listdir(os.getcwd())
            for sub in subdirs:
                if " " in sub:
                    os.rename(sub, sub.replace(" ", "_"))

            subdirs = os.listdir(os.getcwd())
            for sub in subdirs:
                if os.path.isdir(sub):
                    os.chdir(sub)
                    print("\nGenerating XML for:", sub)

                    os.chdir("Label")
                    for file in tqdm(os.listdir(os.getcwd())):
                        if file.endswith(".txt"):
                            image_id = file.split(".")[0]
                            xml_root = etree.Element("annotation")

                            os.chdir("..")
                            folder_el = etree.Element("folder")
                            folder_el.text = os.path.basename(os.getcwd())
                            xml_root.append(folder_el)

                            file_el = etree.Element("filename")
                            file_el.text = image_id + ".jpg"
                            xml_root.append(file_el)

                            path_el = etree.Element("path")
                            path_el.text = os.path.join(os.path.dirname(os.path.abspath(file)), image_id + ".jpg")
                            xml_root.append(path_el)

                            source_el = etree.Element("source")
                            db_el = etree.Element("database")
                            db_el.text = "Unknown"
                            source_el.append(db_el)
                            xml_root.append(source_el)

                            size_el = etree.Element("size")
                            image = cv2.imread(file_el.text)
                            try:
                                width_el = etree.Element("width")
                                width_el.text = str(image.shape[1])
                            except AttributeError:
                                os.chdir("Label")
                                continue

                            height_el = etree.Element("height")
                            height_el.text = str(image.shape[0])
                            depth_el = etree.Element("depth")
                            depth_el.text = str(image.shape[2])

                            size_el.extend([width_el, height_el, depth_el])
                            xml_root.append(size_el)

                            segmented_el = etree.Element("segmented")
                            segmented_el.text = "0"
                            xml_root.append(segmented_el)

                            os.chdir("Label")
                            with open(file, 'r') as label_data:
                                for line in label_data:
                                    data = line.strip().split(' ')
                                    cls_parts = len(data) - 4
                                    cls_name = data[0]
                                    for i in range(1, cls_parts):
                                        cls_name = f"{cls_name}_{data[i]}"

                                    offset = cls_parts
                                    xmin_val = str(int(round(float(data[0 + offset]))))
                                    ymin_val = str(int(round(float(data[1 + offset]))))
                                    xmax_val = str(int(round(float(data[2 + offset]))))
                                    ymax_val = str(int(round(float(data[3 + offset]))))

                                    obj_el = etree.Element("object")

                                    name_el = etree.Element("name")
                                    name_el.text = cls_name
                                    obj_el.append(name_el)

                                    pose_el = etree.Element("pose")
                                    pose_el.text = "Unspecified"
                                    obj_el.append(pose_el)

                                    truncated_el = etree.Element("truncated")
                                    truncated_el.text = "0"
                                    obj_el.append(truncated_el)

                                    difficult_el = etree.Element("difficult")
                                    difficult_el.text = "0"
                                    obj_el.append(difficult_el)

                                    box_el = etree.Element("bndbox")
                                    for coord, val in zip(["xmin", "ymin", "xmax", "ymax"], [xmin_val, ymin_val, xmax_val, ymax_val]):
                                        coord_el = etree.Element(coord)
                                        coord_el.text = val
                                        box_el.append(coord_el)

                                    obj_el.append(box_el)
                                    xml_root.append(obj_el)

                            os.chdir("..")
                            xml_output = etree.tostring(xml_root, pretty_print=True)
                            with open(image_id + ".xml", 'wb') as out_file:
                                out_file.write(xml_output)

                            os.chdir("Label")

                    os.chdir("..")
                    os.chdir("..")
            os.chdir("..")

run_conversion()
#