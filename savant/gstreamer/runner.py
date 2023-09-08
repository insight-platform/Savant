"""GStreamer pipeline runner class."""
import os
import threading
from datetime import timedelta
from pathlib import Path
from time import time
from typing import Optional, Union

from gi.repository import GLib, Gst  # noqa:F401

from savant.healthcheck.status import PipelineStatus
from savant.utils.logging import get_logger

from .pipeline import GstPipeline

logger = get_logger(__name__)


class StateChangeError(Exception):
    """Gst.StateChangeReturn.FAILURE Exception."""


class GstPipelineRunner:
    """Manages running Gstreamer pipeline.

    :param pipeline: GstPipeline or Gst.Pipeline to run.
    :param status_filepath: Path to status file.
    """

    def __init__(
        self,
        pipeline: Union[GstPipeline, Gst.Pipeline],
        status_filepath: Optional[Path] = None,
    ):
        self._status_filepath = status_filepath

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

        self._pipeline: Union[GstPipeline, Gst.Pipeline] = pipeline
        self._gst_pipeline: Gst.Pipeline = (
            pipeline.pipeline if isinstance(pipeline, GstPipeline) else pipeline
        )

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
        self._set_pipeline_status(PipelineStatus.STARTING)

        bus = self._pipeline.get_bus()
        logger.debug('Adding signal watch and connecting callbacks...')
        bus.add_signal_watch()
        bus.connect('message::error', self.on_error)
        bus.connect('message::eos', self.on_eos)
        bus.connect('message::warning', self.on_warning)
        bus.connect('message::state-changed', self.on_state_changed)

        logger.debug('Setting pipeline to READY...')
        self._pipeline.set_state(Gst.State.READY)

        logger.debug('Setting pipeline to PLAYING...')
        self._pipeline.set_state(Gst.State.PLAYING)

        if isinstance(self._pipeline, GstPipeline):
            logger.debug('Calling pipeline.on_startup()...')
            self._pipeline.on_startup()

        logger.debug('Starting main loop thread...')
        self._is_running = True
        self._main_loop_thread.start()

        end_time = time()
        exec_seconds = end_time - start_time
        logger.info(
            'The pipeline is initialized and ready to process data. Initialization took %s.',
            timedelta(seconds=exec_seconds),
        )

        self._start_time = end_time
        self._set_pipeline_status(PipelineStatus.RUNNING)

    def shutdown(self):
        """Stops pipeline."""
        logger.debug('shutdown() called.')
        if not self._is_running:
            logger.debug('The pipeline is shutting down already.')
            return

        self._set_pipeline_status(PipelineStatus.STOPPING)
        self._is_running = False

        if isinstance(self._pipeline, GstPipeline):
            logger.debug('Calling pipeline.before_shutdown()...')
            self._pipeline.before_shutdown()

        if self._main_loop.is_running():
            logger.debug('Quitting main loop...')
            self._main_loop.quit()

        logger.debug('Setting pipeline to NULL...')
        self._pipeline.set_state(Gst.State.NULL)

        exec_seconds = time() - self._start_time
        logger.info(
            'The pipeline is about to stop. Operation took %s.',
            timedelta(seconds=exec_seconds),
        )

        if isinstance(self._pipeline, GstPipeline):
            logger.debug('Calling pipeline.on_shutdown()...')
            self._pipeline.on_shutdown()

        self._set_pipeline_status(PipelineStatus.STOPPED)

    def on_error(  # pylint: disable=unused-argument
        self, bus: Gst.Bus, message: Gst.Message
    ):
        """Error callback."""
        err, debug = message.parse_error()
        # calling `raise` here causes the pipeline to hang,
        # just save message and handle it later
        self._error = self.build_error_message(message, err, debug)
        logger.error(self._error)
        self._error += f' Debug info: "{debug}".'
        self.shutdown()

    def build_error_message(self, message: Gst.Message, err: GLib.GError, debug: str):
        """Build error message."""
        return f'Received error "{err}" from {message.src.name}.'

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

    def on_state_changed(  # pylint: disable=unused-argument
        self, bus: Gst.Bus, msg: Gst.Message
    ):
        """Change state callback."""
        if not msg.src == self._gst_pipeline:
            # not from the pipeline, ignore
            return

        old_state, new_state, _ = msg.parse_state_changed()
        old_state_name = Gst.Element.state_get_name(old_state)
        new_state_name = Gst.Element.state_get_name(new_state)
        logger.debug(
            'Pipeline state changed from %s to %s.', old_state_name, new_state_name
        )

        if old_state != new_state and os.getenv('GST_DEBUG_DUMP_DOT_DIR'):
            file_name = f'pipeline.{old_state_name}_{new_state_name}'
            Gst.debug_bin_to_dot_file_with_ts(
                self._gst_pipeline, Gst.DebugGraphDetails.ALL, file_name
            )

    def _set_pipeline_status(self, status: PipelineStatus):
        if self._status_filepath is not None:
            logger.info('Setting pipeline status to %s.', status)
            self._status_filepath.write_text(f'{status.value}\n')
