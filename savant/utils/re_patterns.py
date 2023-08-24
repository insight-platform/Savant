import re

socket_uri_pattern = re.compile('([a-z]+\\+[a-z]+:)?([a-z]+://.*)')
socket_options_pattern = re.compile('([a-z]+)\\+([a-z]+):')
