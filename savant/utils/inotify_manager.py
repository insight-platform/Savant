from collections import defaultdict
import logging
from inotify_simple import INotify, flags
from savant.utils.singleton import SingletonMeta


class INotifyManager(metaclass=SingletonMeta):
    def __init__(self) -> None:
        self.logger = logging.getLogger(__name__)
        self.inotify = INotify()
        self.watch_flags = flags.MODIFY | flags.DELETE_SELF | flags.CLOSE_WRITE

        self.file_to_watch = {}
        self.subscriber_to_watch = {}
        self.watch_to_subscribers = defaultdict(list)
        self.replies = defaultdict(bool)

    def add_watch(self, file_path, subscriber):

        if file_path in self.file_to_watch:
            self.logger.debug('%s: %s is already watched.', subscriber, file_path)
            watch_descriptor = self.file_to_watch[file_path]
        else:
            # watches are supposed to be freed automatically
            # when the process exits
            # https://man7.org/linux/man-pages/man7/inotify.7.html
            watch_descriptor = self.inotify.add_watch(file_path, self.watch_flags)
            self.file_to_watch[file_path] = watch_descriptor
            self.logger.debug(
                '%s: added watch %s for %s.', subscriber, watch_descriptor, file_path
            )

        self.subscriber_to_watch[subscriber] = watch_descriptor
        self.watch_to_subscribers[watch_descriptor].append(subscriber)

    def is_changed(self, subscriber):

        watch_descriptor = self.subscriber_to_watch[subscriber]
        self.logger.debug('%s: checking watch %s.', subscriber, watch_descriptor)

        if self.replies[subscriber]:
            self.logger.debug(
                '%s: prev reply %s.', subscriber, self.replies[subscriber]
            )
            self.replies[subscriber] = False
            return True

        for event in self.inotify.read(timeout=0):
            if event.mask & flags.MODIFY:
                self.logger.debug('MODIFY event found for watch %s.', event.wd)
                for subscriber in self.watch_to_subscribers[event.wd]:
                    self.logger.debug('%s: save True reply.', subscriber)
                    self.replies[subscriber] = True

        reply = self.replies[subscriber]
        self.replies[subscriber] = False
        return reply
