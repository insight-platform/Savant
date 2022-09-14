"""Model registry."""
from typing import Any, Dict, List, Tuple, Optional
from savant.utils.singleton import SingletonMeta


class ModelObjectRegistry(metaclass=SingletonMeta):
    """Model.Object registry.

    Stores model+object with relevant model_uid+class_id to manage model
    dependencies (operate-on).
    """

    REGISTRY_KEY_SEPARATOR: str = '.'

    def __init__(self):
        # {'model_name.object_label': (model_uid, object_class_id)}
        self._object_registry: Dict[str, Tuple[int, int]] = {}
        # {'model_name': model_uid}
        self._model_registry: Dict[str, int] = {}
        self._uid2name: Dict[int, str] = {}
        # model_uid sequential generation
        self._model_uid = 0

    def __str__(self):
        return f'{self._model_registry}, {self._object_registry}'

    @property
    def new_model_uid(self):
        """Generate new model uid."""
        self._model_uid += 1
        return self._model_uid

    @staticmethod
    def model_object_key(model_name: str, object_label: str) -> str:
        """Returns unique key for specified model object type, used in the
        registry and in NvDsObjectMeta.obj_label value."""

        if not model_name:  # frame object
            return object_label
        return f'{model_name}{ModelObjectRegistry.REGISTRY_KEY_SEPARATOR}{object_label}'

    @staticmethod
    def parse_model_object_key(model_object_key: str) -> Tuple[str, str]:
        """Parses model object type key, returns model name and object
        label."""

        model_name, object_label = '', model_object_key
        if ModelObjectRegistry.REGISTRY_KEY_SEPARATOR in model_object_key:
            model_name, object_label = model_object_key.split(
                ModelObjectRegistry.REGISTRY_KEY_SEPARATOR
            )
        return model_name, object_label

    def register_model(
        self,
        model_element_name: str,
        model_output_object_labels: Optional[Dict[int, str]] = None,
    ) -> int:
        """Register a model element."""
        model_uid = self.new_model_uid
        self._model_registry[model_element_name] = model_uid
        self._uid2name[model_uid] = model_element_name
        # register model objects
        if model_output_object_labels:
            for class_id, label in model_output_object_labels.items():
                self._object_registry[
                    self.model_object_key(model_element_name, label)
                ] = (model_uid, class_id)
        return model_uid

    def get_name(self, uid: int) -> str:
        """Get model name from uid."""
        return self._uid2name[uid]

    def get_model_uid(self, model_name: str) -> int:
        """Get model uid from name."""
        return self._model_registry[model_name]

    def is_model_object_key_registered(self, model_object_key: str) -> bool:
        """Check if model object key is registered."""
        return model_object_key in self._object_registry

    def get_model_object_ids(
        self,
        model_object_key: Optional[str] = None,
        model_name: Optional[str] = None,
        object_label: Optional[str] = None,
    ) -> Tuple[int, int]:
        """Returns tuple(model_uid, class_id) for specified key or specified
        model object type. Adds new record if there is no model info in the
        registry.

        :param model_object_key:
        :param model_name:
        :param object_label:
        :return: (model_uid, class_id)
        """
        assert model_object_key is not None or (
            model_name is not None and object_label is not None
        )

        if model_object_key is None:
            model_object_key = ModelObjectRegistry.model_object_key(
                model_name, object_label
            )

        if model_object_key not in self._object_registry:
            model_name, object_label = self.parse_model_object_key(model_object_key)
            model_uid, class_id = None, 0
            # try to find model_uid for model first
            if model_name:
                model_class_ids = [
                    v
                    for k, v in self._object_registry.items()
                    if k.startswith(
                        f'{model_name}{ModelObjectRegistry.REGISTRY_KEY_SEPARATOR}'
                    )
                ]
                if model_class_ids:
                    model_uid = model_class_ids[0][0]
                    class_id = max(class_id for _, class_id in model_class_ids) + 1

            if not model_uid:
                try:
                    model_uid = self.get_model_uid(model_name)
                except KeyError:
                    model_uid = self.register_model(model_name)

            self._object_registry[model_object_key] = model_uid, class_id

        return self._object_registry[model_object_key]
