"""Inotify manager module."""
import logging
import sys
from collections import defaultdict
from importlib import util as importlib_util
from importlib.machinery import ModuleSpec
from types import ModuleType
from typing import Dict, Optional

from inotify_simple import INotify, flags

from savant.utils.logging import get_logger
from savant.utils.singleton import SingletonMeta

logger = get_logger(__name__)


def import_module(spec: ModuleSpec) -> ModuleType:
    """Instantiate a module from module spec."""
    module_instance = importlib_util.module_from_spec(spec)
    spec.loader.exec_module(module_instance)
    # reloading requires module being put into the modules dict
    # only do so after the loader.exec_module call as it can fail
    sys.modules[spec.name] = module_instance
    return module_instance


class ModulesCache(metaclass=SingletonMeta):
    """Singleton facilitates caching and reloading of user updates
    PyFunc modules in dev mode.
    """

    def __init__(self) -> None:
        self.logger = get_logger(__name__)
        self.inotify = INotify()
        self.watch_flags = flags.MODIFY

        self.module_file_to_watch = {}
        self.watch_to_module_file: Dict[str, str] = {}
        self.changed_modules = defaultdict(bool)
        self._cache_module: Dict[str, ModuleType] = {}

    def get_module_instance(self, module_spec: ModuleSpec) -> Optional[ModuleType]:
        """Get module instance by module specification.

        :param module_spec: user module specification.
        :return: module instance.
        """
        module_file = module_spec.origin
        if module_file not in self._cache_module:
            logger.debug(f'{module_file} is not cached, importing.')
            if module_spec is None:
                logger.warning('No module spec, caching is impossible.')
                return None
            elif not module_spec.has_location:
                logger.warning(
                    'Module spec with undetermined location, '
                    'caching and reloading is impossible'
                )
                return None
            self._cache_module[module_file] = import_module(module_spec)
            self.add_watch(module_file)
        elif self._is_file_changed(module_file):
            self.logger.debug('File `%s` changed, reload module.', module_file)
            self._cache_module[module_file] = import_module(module_spec)
            self.changed_modules[module_file] = False
        return self._cache_module[module_file]

    def add_watch(self, module_file: str):
        """Add a watch for the module file.

        :param module_file: path to module file.
        """
        if module_file in self.module_file_to_watch:
            watch_descriptor = self.module_file_to_watch[module_file]
            self.logger.debug(
                'Watch descriptor %s already watches module (%s)',
                watch_descriptor,
                module_file,
            )
            return

        # watches are supposed to be freed automatically
        # when the process exits
        # https://man7.org/linux/man-pages/man7/inotify.7.html
        watch_descriptor = self.inotify.add_watch(module_file, self.watch_flags)
        self.logger.debug('Added watch %s for file %s.', watch_descriptor, module_file)

        self.module_file_to_watch[module_file] = watch_descriptor
        self.watch_to_module_file[watch_descriptor] = module_file

    def _is_file_changed(self, module_file: str) -> bool:
        for event in self.inotify.read(timeout=0):
            if event.mask & self.watch_flags:
                self.logger.debug('MODIFY event found for watch %s.', event.wd)
                module_file = self.watch_to_module_file[event.wd]
                self.logger.debug('%s: save True reply.', module_file)
                self.changed_modules[module_file] = True
        logger.debug(
            '%s is changed: %s', module_file, self.changed_modules[module_file]
        )
        return self.changed_modules[module_file]

    def is_changed(self, user_module: ModuleType) -> bool:
        """Check existing watches, return True if the file the subscriber watches
        has changed since last call.

        :param user_module: instance of the user module.
        :return: True if the watched file has changed, False otherwise.
        """
        module_file = user_module.__spec__.origin

        if self.logger.isEnabledFor(logging.DEBUG):
            watch_descriptor = self.module_file_to_watch[module_file]
            self.logger.debug(
                '%s: checking watch %s.',
                module_file,
                watch_descriptor,
            )
        file_is_changed = self._is_file_changed(module_file)

        return file_is_changed or user_module is not self._cache_module[module_file]
