"""Inotify manager module."""
import logging
from collections import defaultdict

from inotify_simple import INotify, flags

from savant.utils.singleton import SingletonMeta


class INotifyManager(metaclass=SingletonMeta):
    """Singleton facilitates monitoring changes in files from pyfuncs
    using inotify python bindinds package.
    """

    def __init__(self) -> None:
        self.logger = logging.getLogger(__name__)
        self.inotify = INotify()
        self.watch_flags = flags.MODIFY

        self.file_to_watch = {}
        self.watch_to_file = {}
        self.subscriber_to_watch = {}
        self.watch_to_subscribers = defaultdict(list)
        self.replies = defaultdict(bool)

    def add_watch(self, file_path: str, subscriber: int):
        """Add a watch for the specified file and subscriber.

        :param file_path: which file to watch for changes.
        :param subscriber: pyfunc's id.
        """
        if subscriber in self.subscriber_to_watch:
            watch_descriptor = self.subscriber_to_watch[subscriber]
            file_path = self.watch_to_file[watch_descriptor]
            self.logger.debug(
                '%s: subscriber already watches %s.', subscriber, file_path
            )
            return

        if file_path in self.file_to_watch:
            self.logger.debug('%s: %s is already watched.', subscriber, file_path)
            watch_descriptor = self.file_to_watch[file_path]
        else:
            # watches are supposed to be freed automatically
            # when the process exits
            # https://man7.org/linux/man-pages/man7/inotify.7.html
            watch_descriptor = self.inotify.add_watch(file_path, self.watch_flags)
            self.file_to_watch[file_path] = watch_descriptor
            self.watch_to_file[watch_descriptor] = file_path
            self.logger.debug(
                '%s: added watch %s for %s.', subscriber, watch_descriptor, file_path
            )

        self.subscriber_to_watch[subscriber] = watch_descriptor
        self.watch_to_subscribers[watch_descriptor].append(subscriber)

    def is_changed(self, subscriber: int) -> bool:
        """Check existing watches, return True if the file the subscriber watches
        has changed since last call.

        :param subscriber: pyfunc's id.
        :return: True if the watched file has changed, False otherwise.
        """
        watch_descriptor = self.subscriber_to_watch[subscriber]
        self.logger.debug(
            '%s: checking watch %s.',
            subscriber,
            watch_descriptor,
        )

        if self.replies[subscriber]:
            self.logger.debug(
                '%s: prev reply %s.', subscriber, self.replies[subscriber]
            )
            self.replies[subscriber] = False
            return True

        for event in self.inotify.read(timeout=0):
            if event.mask & self.watch_flags:
                self.logger.debug('MODIFY event found for watch %s.', event.wd)
                for sub_iter in self.watch_to_subscribers[event.wd]:
                    self.logger.debug('%s: save True reply.', sub_iter)
                    self.replies[sub_iter] = True

        reply = self.replies[subscriber]
        self.replies[subscriber] = False
        return reply
