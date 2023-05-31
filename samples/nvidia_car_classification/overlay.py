"""Draw func adds car classification models outputs: car color, car make, car type."""

from savant.deepstream.drawfunc import NvDsDrawFunc
from savant.meta.object import ObjectMeta


class Overlay(NvDsDrawFunc):

    def override_draw_spec(self, object_meta: ObjectMeta, draw_spec_cfg: dict) -> dict:
        # add classifier attributes labels to the object visualisation
        # one attribute per line
        attr_meta = object_meta.get_attr_meta("Secondary_CarColor", 'car_color')
        if attr_meta is not None:
            draw_spec_cfg['label']['format'] += [str(attr_meta.value)]

        attr_meta = object_meta.get_attr_meta("Secondary_CarMake", 'car_make')
        if attr_meta is not None:
            draw_spec_cfg['label']['format'] += [str(attr_meta.value)]

        attr_meta = object_meta.get_attr_meta("Secondary_VehicleTypes", 'car_type')
        if attr_meta is not None:
            draw_spec_cfg['label']['format'] += [str(attr_meta.value)]

        # don't forget to return the modified draw spec
        return draw_spec_cfg
