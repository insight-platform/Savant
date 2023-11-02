import sys
from itertools import permutations

import yaml
from savant_rs.primitives.geometry import Point, PolygonalArea

from savant.deepstream.meta.frame import NvDsFrameMeta
from savant.deepstream.pyfunc import NvDsPyFuncPlugin
from savant.gstreamer import Gst


class ConditionalDetectorSkip(NvDsPyFuncPlugin):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        with open(self.config_path, 'r', encoding='utf8') as stream:
            self.area_config = yaml.safe_load(stream)

    def process_frame(self, buffer: Gst.Buffer, frame_meta: NvDsFrameMeta):
        primary_meta_object = None
        for obj_meta in frame_meta.objects:
            if obj_meta.is_primary:
                primary_meta_object = obj_meta
                break

        # if the boundary lines are not configured for this source
        # then disable detector inference entirely by removing the primary object
        # Note:
        # In order to enable use cases such as conditional inference skip
        # or user-defined ROI, Savant configures all Deepstream models to run
        # in 'secondary' mode and inserts a primary 'frame' object into the DS meta
        if (
            primary_meta_object is not None
            and frame_meta.source_id not in self.area_config
        ):
            frame_meta.remove_obj_meta(primary_meta_object)


class ObjectCounting(NvDsPyFuncPlugin):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        with open(self.config_path, 'r', encoding='utf8') as stream:
            self.area_config = yaml.safe_load(stream)

        self.areas = {}
        for source_id, areas in self.area_config.items():
            self.areas[source_id] = {}
            for area_name, area_dict in areas.items():

                coords_list = area_dict['points']

                points = [Point(*pt_coords) for pt_coords in coords_list]

                points_permutations = permutations(points)
                config_order = next(points_permutations)

                polygon = PolygonalArea(config_order)
                if polygon.is_self_intersecting():
                    # try to find a permutation of points that does not produce a self-intersecting polygon
                    self.logger.warning(
                        'Polygon config for the "%s" source id produced a self-intersecting polygon '
                        'for the "%s" area.'
                        ' Trying to find a valid permutation...',
                        source_id,
                        area_name,
                    )
                    while True:
                        try:
                            points_perm = next(points_permutations)
                            polygon = PolygonalArea(points_perm)
                            if not polygon.is_self_intersecting():
                                self.logger.info(
                                    'Found a valid points permutation "%s" for the "%s" source id "%s" area.',
                                    points_perm,
                                    source_id,
                                    area_name,
                                )
                                break
                        except StopIteration:
                            self.logger.error(
                                'Polygon config for the "%s" source id produced a self-intersecting polygon.'
                                ' Please correct coordinates of "%s" area in the config file and restart the pipeline.',
                                source_id,
                                area_name,
                            )
                            sys.exit(1)

                self.areas[source_id][area_name] = polygon

    def process_frame(self, buffer: Gst.Buffer, frame_meta: NvDsFrameMeta):
        """Process frame metadata.

        :param buffer: Gstreamer buffer with this frame's data.
        :param frame_meta: This frame's metadata.
        """

        primary_meta_object = None
        obj_metas = []
        for obj_meta in frame_meta.objects:
            if obj_meta.is_primary:
                primary_meta_object = obj_meta
            elif obj_meta.label == self.target_obj_label:
                obj_metas.append(obj_meta)

        if not primary_meta_object:
            return

        obj_centers = [Point(obj.bbox.xc, obj.bbox.yc) for obj in obj_metas]

        areas = self.areas.get(frame_meta.source_id, [])
        for area_name, area in areas.items():
            contain_flags = area.contains_many_points(obj_centers)
            for obj, is_in_zone in zip(obj_metas, contain_flags):
                if is_in_zone:
                    obj.draw_label = area_name
            n_objs_in_area = sum(contain_flags)

            primary_meta_object.add_attr_meta('analytics', area_name, n_objs_in_area)
