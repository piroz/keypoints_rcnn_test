from genericpath import exists
import json
import glob
import os
import shutil

def convert_coco(annotation):
    bboxes = []
    keypoints = []
    bbox = annotation["bbox"]
    bboxes.append([bbox[0], bbox[1], bbox[0] + bbox[2], bbox[1] + bbox[3]])
    k_list = annotation["keypoints"]
    k_list = [k_list[i:i + 3] for i in range(0, len(k_list), 3)]
    keypoints.append(k_list)

    return {
        "bboxes": bboxes,
        "keypoints": keypoints
    }

def import_coco_annotator(json_path):
    saved_path = os.path.dirname(json_path)
    train_image_dir = "{}/train/images".format(saved_path)
    train_annotations_dir = "{}/train/annotations".format(saved_path)
    test_image_dir = "{}/test/images".format(saved_path)
    test_annotations_dir = "{}/test/annotations".format(saved_path)

    for output_path in [train_image_dir, train_annotations_dir, test_image_dir, test_annotations_dir]:
        os.makedirs(output_path, exist_ok=True)

    with open(json_path) as j:
        exported_data = json.load(j)

    for i, image in enumerate(exported_data["images"]):
        image_file_name = image["file_name"]
        image_file_exclude_ext = os.path.splitext(image_file_name)[0]
        json_file_name = "{}.json".format(image_file_exclude_ext)
        anotation = convert_coco(exported_data["annotations"][i])
        if i % 5 == 0:
            with open("{}/{}".format(test_annotations_dir, json_file_name), "w") as out:
                json.dump(anotation, out)
                shutil.copy("{}/{}".format(saved_path, image_file_name), "{}/{}.jpg".format(test_image_dir, image_file_exclude_ext))
        else:
            with open("{}/{}".format(train_annotations_dir, json_file_name), "w") as out:
                json.dump(anotation, out)
                shutil.copy("{}/{}".format(saved_path, image_file_name), "{}/{}.jpg".format(train_image_dir, image_file_exclude_ext))

def main():
    json_path = "./coco_annotator_saved/exported.json"
    if not os.path.exists(json_path):
        raise RuntimeError("`{}` does not exists.".format(json_path))
    import_coco_annotator(json_path)

if __name__ == "__main__":
    main()