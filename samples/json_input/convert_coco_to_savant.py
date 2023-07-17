import json
from functools import partial
from pathlib import Path
import imagesize
import click
from pycocotools.coco import COCO
import numpy as np

# from savant.meta.constants import UNTRACKED_OBJECT_ID


@click.command()
@click.option('--annotation_folder', required=True, help="Folder containing the coco annotation files")
@click.option('--image_folder', required=True, help="Folder containing the coco image files")
@click.option('--output_folder', required=True, help="Folder to save the savant annotation files")
def main(
    annotation_folder,
    image_folder: str,
    output_folder: str
):
    # Read the coco file
    annotation_folder = Path(annotation_folder)
    image_folder = Path(image_folder)
    output_folder = Path(output_folder)
    pts = 0
    files_list = list(annotation_folder.glob('*.txt'))
    files_list = sorted(files_list, key=lambda x: x.name)

    coco = COCO(annotation_folder / "annotations" / f"instances_val2017.json")
    cat_ids = coco.getCatIds()
    for file in files_list:
        img_info = coco.loadImgs(int(file.stem))[0]
        ann_ids = coco.getAnnIds(imgIds=img_info['id'], catIds=cat_ids, iscrowd=False)
        anns = coco.loadAnns(ann_ids)
        objects_savant = []
        for obj in anns:
            bbox = obj["bbox"]
            x = round(bbox[0])
            y = round(bbox[1])
            width = round(bbox[2])
            height = round(bbox[3])
            label = coco.cats[obj["category_id"]]["name"]
            obj_savant = dict(
                model_name="coco",
                label=label,
                object_id=1,
                bbox=dict(
                    xc=x + width/2,
                    yc=y + height/2,
                    width=width,
                    height=height,
                    angle=0.0
                ),
                confidence=1,
                attributes=[],
                parent_model_name=None,
                parent_label=None,
                parent_object_id=None
            )
            objects_savant.append(obj_savant)
        output_dict = dict(
                pts=pts,
                metadata=dict(objects=objects_savant),
                schema="VideoFrame",
            )
        pts += 33333333
        json.dump(output_dict, open(output_folder / f"{file.stem}.json", 'w'))

def coco_to_rel_savant(coco_line, image_width, image_height, coco_label):
    """Convert a coco format to a savant bbox format"""
    split_line = coco_line.split(" ")
    class_id = int(split_line[0])
    coords = list(map(float, split_line[1:]))
    left = min(coords[0::2])
    top = min(coords[1::2])
    right = max(coords[0::2])
    bottom = max(coords[1::2])
    return dict(
        model_name="coco",
        label=coco_label[class_id],
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
