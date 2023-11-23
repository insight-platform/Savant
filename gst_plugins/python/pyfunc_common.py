import inspect
import json
from logging import Logger
from typing import Any, Optional

from savant.base.pyfunc import PyFunc, PyFuncNoopCallException
from savant.gstreamer import Gst
from savant.gstreamer.utils import (
    gst_post_stream_failed_error,
    gst_post_stream_failed_warning,
)


def init_pyfunc(
    gst_element: Gst.Element,
    logger: Logger,
    module: Optional[str],
    class_name: Optional[str],
    kwargs: Optional[str],
    dev_mode: bool = False,
) -> PyFunc:
    # pylint: disable=broad-exception-caught
    if kwargs:
        try:
            kwargs = json.loads(kwargs)
        except Exception as exc:
            return handle_fatal_error(
                gst_element,
                logger,
                exc,
                f'Failed to parse kwargs for "{module}.{class_name}" pyfunc.',
                dev_mode,
                True,
                False,
            )
    else:
        kwargs = None

    try:
        pyfunc = PyFunc(
            module=module,
            class_name=class_name,
            kwargs=kwargs,
            dev_mode=dev_mode,
        )
    except Exception as exc:
        return handle_fatal_error(
            gst_element,
            logger,
            exc,
            f'Failed to initialize "{module}.{class_name}" pyfunc.',
            dev_mode,
            True,
            False,
        )

    try:
        pyfunc.load_user_code()
    except Exception as exc:
        return handle_fatal_error(
            gst_element,
            logger,
            exc,
            f'Failed to load user code for {pyfunc}.',
            dev_mode,
            True,
            False,
        )

    return pyfunc


def handle_fatal_error(
    gst_element: Gst.Element,
    logger: Logger,
    exc: BaseException,
    msg: str,
    dev_mode: bool,
    return_ok: Any = Gst.FlowReturn.OK,
    return_err: Any = Gst.FlowReturn.ERROR,
) -> Any:
    if dev_mode:
        if not isinstance(exc, PyFuncNoopCallException):
            logger.exception(msg)
        return return_ok

    gst_post_stream_failed_error(
        gst_element=gst_element,
        frame=inspect.currentframe(),
        file_path=__file__,
        text=msg,
    )
    logger.exception(msg)
    return return_err


def handle_non_fatal_error(
    gst_element: Gst.Element,
    logger: Logger,
    exc: BaseException,
    msg: str,
    dev_mode: bool,
):
    if dev_mode:
        if not isinstance(exc, PyFuncNoopCallException):
            logger.warning(msg)
        return

    gst_post_stream_failed_warning(
        gst_element=gst_element,
        frame=inspect.currentframe(),
        file_path=__file__,
        text=msg,
    )
    logger.warning(msg, exc_info=exc)
