"""GStreamer pipeline runner class."""
from datetime import timedelta

from time import time
from typing import Optional
import logging
import threading
from gi.repository import GLib, Gst  # noqa:F401
from .pipeline import GstPipeline

logger = logging.getLogger(__name__)


class StateChangeError(Exception):
    """Gst.StateChangeReturn.FAILURE Exception."""


class GstPipelineRunner:
    """Manages running Gstreamer pipeline.

    :param pipeline: GstPipeline to run.
    """

    def __init__(self, pipeline: GstPipeline):
        # pipeline error storage
        self._error: Optional[str] = None

        # running pipeline flag
        self._is_running = False

        # pipeline execution start time, will be set on startup
        self._start_time = 0.0

        # pipeline event loop
        # alternative: bus.timed_pop_filtered(
        #   Gst.CLOCK_TIME_NONE, Gst.MessageType.ERROR | Gst.MessageType.EOS)
        # TODO: Do we really need event loop in thread and queue here?

        self._main_loop = GLib.MainLoop()
        self._main_loop_thread = threading.Thread(target=self._main_loop_run)

        self._pipeline: GstPipeline = pipeline

    def __enter__(self):
        self.startup()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.shutdown()

    def _main_loop_run(self):
        try:
            self._main_loop.run()
        finally:
            self.shutdown()
            if self._error:
                raise RuntimeError(self._error)

    def startup(self):
        """Starts pipeline."""
        logger.info('Starting pipeline `%s`...', self._pipeline)
        start_time = time()

        bus = self._pipeline.get_bus()
        logger.debug('Adding signal watch and connecting callbacks...')
        bus.add_signal_watch()
        bus.connect('message::error', self.on_error)
        bus.connect('message::eos', self.on_eos)
        bus.connect('message::warning', self.on_warning)

        logger.debug('Setting pipeline to READY...')
        self._pipeline.set_state(Gst.State.READY)

        logger.debug('Setting pipeline to PLAYING...')
        self._pipeline.set_state(Gst.State.PLAYING)

        logger.debug('Calling pipeline.on_startup()...')
        self._pipeline.on_startup()

        logger.debug('Starting main loop thread...')
        self._is_running = True
        self._main_loop_thread.start()

        end_time = time()
        exec_seconds = end_time - start_time
        logger.info(
            'Pipeline starting ended after %s.', timedelta(seconds=exec_seconds)
        )

        self._start_time = end_time

    def shutdown(self):
        """Stops pipeline."""
        logger.debug('shutdown() called.')
        if not self._is_running:
            return

        self._is_running = False

        if self._main_loop.is_running():
            logger.debug('Quitting main loop...')
            self._main_loop.quit()

        logger.debug('Setting pipeline to NULL...')
        self._pipeline.set_state(Gst.State.NULL)

        exec_seconds = time() - self._start_time
        logger.info(
            'Pipeline execution ended after %s.', timedelta(seconds=exec_seconds)
        )

        logger.debug('Calling pipeline.on_shutdown()...')
        self._pipeline.on_shutdown()

    def on_error(  # pylint: disable=unused-argument
        self, bus: Gst.Bus, message: Gst.Message
    ):
        """Error callback."""
        logger.debug('Error callback.')
        err, debug = message.parse_error()
        # calling `raise` here causes the pipeline to hang,
        # just save message and handle it later
        self._error = f'Received error {err} from {message.src.name}. {debug}.'
        self.shutdown()

    def on_eos(  # pylint: disable=unused-argument
        self, bus: Gst.Bus, message: Gst.Message
    ):
        """EOS callback."""
        logger.info('Received EOS from %s.', message.src.name)
        self.shutdown()

    def on_warning(  # pylint: disable=unused-argument
        self, bus: Gst.Bus, message: Gst.Message
    ):
        """Warning callback."""
        warn, debug = message.parse_warning()
        logger.warning('Received warning %s. %s', warn, debug)
