import json
from functools import partial
from pathlib import Path
import imagesize
import click
import numpy as np

# from savant.meta.constants import UNTRACKED_OBJECT_ID


@click.command()
@click.option('--annotation_folder', required=True, help="Folder containing the coco annotation files")
@click.option('--image_folder', required=True, help="Folder containing the coco image files")
@click.option('--output_folder', required=True, help="Folder to save the savant annotation files")
def main(
    annotation_folder,
    image_folder:str,
    output_folder: str
):
    # Read the coco file
    annotation_folder = Path(annotation_folder)
    image_folder = Path(image_folder)
    output_folder = Path(output_folder)
    pts = 0
    files_list = list(annotation_folder.glob('*.txt'))
    files_list = sorted(files_list, key=lambda x: x.name)
    for file in files_list:
        with open(file, 'r') as fp:
            width, height = imagesize.get(image_folder / f"{file.stem}.jpg")
            print(width, height)
            objects = list(
                map(
                    partial(coco_to_rel_savant, image_width=width, image_height=height),
                    fp.readlines()
                )
            )
            print(objects)

            output_dict = dict(
                # source_id=test-source-people,
                pts=pts,
                # framerate=None,
                # width=width,
                # height=height,
                # dts=null
                # duration=None,
                # "codec": null,
                # "keyframe": true,
                metadata=dict(objects=objects),
                schema="VideoFrame",
            )
            pts+=33333333

        json.dump(output_dict, open(output_folder / f"{file.stem}.json", 'w'))

    # {"source_id": "test-source-people", "pts": 383716666, "framerate": "60000/1001",
    #  "width": 1920, "height": 1080, "dts": null, "duration": 16683333, "codec": null,
    #  "keyframe": true, "metadata": {"objects": [
    #     {"model_name": "peoplenet", "label": "face", "object_id": 0,
    #      "bbox": {"xc": 566.7897338867188, "yc": 628.3839111328125,
    #               "width": 6.2696685791015625, "height": 9.333744049072266,
    #               "angle": 0.0}, "confidence": 0.262939453125, "attributes": [],
    #      "parent_model_name": null, "parent_label": null, "parent_object_id": null},
    #     {"model_name": "peoplenet", "label": "face", "object_id": 1,
    #      "bbox": {"xc": 566.0446166992188, "yc": 626.86767578125,
    #               "width": 10.472488403320312, "height": 12.451114654541016,
    #               "angle": 0.0}, "confidence": 0.279052734375, "attributes": [],
    #      "parent_model_name": null, "parent_label": null, "parent_object_id": null},
    #     {"model_name": "peoplenet", "label": "face", "object_id": 3,
    #      "bbox": {"xc": 1584.3486328125, "yc": 572.4568481445312,
    #               "width": 32.6363525390625, "height": 45.895172119140625,
    #               "angle": 0.0}, "confidence": 0.9951171875, "attributes": [],
    #      "parent_model_name": "peoplenet", "parent_label": "person",
    #      "parent_object_id": 6},
    #     {"model_name": "peoplenet", "label": "person", "object_id": 8,
    #      "bbox": {"xc": 252.98428344726562, "yc": 652.5433349609375,
    #               "width": 50.22045135498047, "height": 134.70919799804688,
    #               "angle": 0.0}, "confidence": 0.61767578125, "attributes": [],
    #      "parent_model_name": null, "parent_label": null, "parent_object_id": null},
    #     {"model_name": "peoplenet", "label": "person", "object_id": 5,
    #      "bbox": {"xc": 1454.7139892578125, "yc": 772.4943237304688,
    #               "width": 111.60186767578125, "height": 417.409423828125,
    #               "angle": 0.0}, "confidence": 0.6455078125, "attributes": [],
    #      "parent_model_name": null, "parent_label": null, "parent_object_id": null},
    #     {"model_name": "peoplenet", "label": "person", "object_id": 6,
    #      "bbox": {"xc": 1595.4493408203125, "yc": 750.1583251953125,
    #               "width": 104.660888671875, "height": 455.53997802734375,
    #               "angle": 0.0}, "confidence": 0.759765625, "attributes": [],
    #      "parent_model_name": null, "parent_label": null, "parent_object_id": null},
    #     {"model_name": "peoplenet", "label": "person", "object_id": 4,
    #      "bbox": {"xc": 1003.5548706054688, "yc": 721.2547607421875,
    #               "width": 133.83755493164062, "height": 323.396484375, "angle": 0.0},
    #      "confidence": 0.86328125, "attributes": [], "parent_model_name": null,
    #      "parent_label": null, "parent_object_id": null},
    #     {"model_name": "auto", "label": "frame", "object_id": 7,
    #      "bbox": {"xc": 960.0, "yc": 540.0000610351562, "width": 1920.0,
    #               "height": 1080.0001220703125, "angle": 0.0},
    #      "confidence": 0.9990000128746033, "attributes": [], "parent_model_name": null,
    #      "parent_label": null, "parent_object_id": null}]},
    #  "tags": {"location": "input_source_frame_meta_people/street_people.mp4"},
    #  "schema": "VideoFrame", "frame_num": 22}

    #     coco = json.load(open(file))
    #     savant = {}
    #     savant['info'] = coco['info']
    #     savant['categories'] = coco['categories']
    #     savant['images'] = coco['images']
    #     savant['annotations'] = coco['annotations']
    #     savant['licenses'] = coco['licenses']
    #
    #     # Write the savant file
    #     savant_file = output_folder / file.name
    #     json.dump(savant, open(savant_file, 'w'))
    # coco_file = sys.argv[1]
    # coco = json.load(open(coco_file))
    #
    # # Convert the coco file to savant format
    # savant = {}
    # savant['info'] = coco['info']
    # savant['categories'] = coco['categories']


def coco_to_rel_savant(coco_line, image_width, image_height):
    """Convert a coco format to a savant bbox format"""
    split_line = coco_line.split(" ")
    class_id = int(split_line[0])
    coords = list(map(float, split_line[1:]))
    left = min(coords[0::2])
    top = min(coords[1::2])
    right = max(coords[0::2])
    bottom = max(coords[1::2])
    return dict(
        model_name="groundtruth",
        label=str(class_id),
        object_id=1,
        bbox=dict(
            xc=(left + right) / 2 * image_width,
            yc=(top + bottom) / 2 * image_height,
            width=(right - left) * image_width,
            height=(bottom - top)*image_height,
            angle=0.0
        ),
        confidence=1,
        attributes=[],
        parent_model_name=None,
        parent_label=None,
        parent_object_id=None
    )


if __name__ == '__main__':
    main()
