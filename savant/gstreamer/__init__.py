"""GStreamer module."""
# to avoid `gi.require_version` warning
import gi

gi.require_version('GLib', '2.0')
gi.require_version('GObject', '2.0')
gi.require_version('Gst', '1.0')
gi.require_version('GstBase', '1.0')
gi.require_version('GstApp', '1.0')
gi.require_version('GstVideo', '1.0')

# pylint:disable=wrong-import-position
from gi.repository import (
    GObject,
    GLib,
    Gst,
    GstBase,
    GstApp,
)
