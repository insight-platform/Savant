import json
from pathlib import Path

import click
from pycocotools.coco import COCO


@click.command()
@click.option(
    '--annotation_folder',
    required=True,
    help='Folder containing the coco annotation files',
)
@click.option(
    '--output_folder', required=True, help='Folder to save the savant annotation files'
)
def main(annotation_folder, output_folder: str):
    # Read the coco file
    annotation_folder = Path(annotation_folder)
    output_folder = Path(output_folder)
    files_list = list(annotation_folder.glob('*.txt'))
    files_list = sorted(files_list, key=lambda file_path: file_path.name)

    coco = COCO(annotation_folder / 'annotations' / f'instances_val2017.json')
    cat_ids = coco.getCatIds()
    for file in files_list:
        img_info = coco.loadImgs(int(file.stem))[0]
        ann_ids = coco.getAnnIds(imgIds=img_info['id'], catIds=cat_ids, iscrowd=False)
        anns = coco.loadAnns(ann_ids)
        objects_savant = []
        object_id = 1
        for obj in anns:
            bbox = obj['bbox']
            x = bbox[0]
            y = bbox[1]
            width = bbox[2]
            height = bbox[3]
            label = coco.cats[obj['category_id']]['name']
            if label == 'person':
                object_id += 1
                obj_savant = dict(
                    model_name='coco',
                    label=label,
                    object_id=object_id,
                    bbox=dict(
                        xc=x + width / 2,
                        yc=y + height / 2,
                        width=width,
                        height=height,
                        angle=0.0,
                    ),
                    confidence=1,
                    attributes=[],
                    parent_model_name=None,
                    parent_label=None,
                    parent_object_id=None,
                )
                objects_savant.append(obj_savant)
        output_dict = dict(metadata=dict(objects=objects_savant))
        json.dump(output_dict, open(output_folder / f'{file.stem}.json', 'w'))


def coco_to_rel_savant(coco_line, image_width, image_height, coco_label):
    """Convert a coco format to a savant input format"""
    split_line = coco_line.split(' ')
    class_id = int(split_line[0])
    coords = list(map(float, split_line[1:]))
    left = min(coords[0::2])
    top = min(coords[1::2])
    right = max(coords[0::2])
    bottom = max(coords[1::2])
    return dict(
        model_name='coco',
        label=coco_label[class_id],
        object_id=1,
        bbox=dict(
            xc=(left + right) / 2 * image_width,
            yc=(top + bottom) / 2 * image_height,
            width=(right - left) * image_width,
            height=(bottom - top) * image_height,
            angle=0.0,
        ),
        confidence=1,
        attributes=[],
        parent_model_name=None,
        parent_label=None,
        parent_object_id=None,
    )


if __name__ == '__main__':
    main()
