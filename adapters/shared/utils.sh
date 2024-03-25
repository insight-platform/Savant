#!/bin/bash

########################
# Print starting message
# Arguments:
#   $1 - the name of starting module or adapter
#########################
function print_starting_message {
  local name=${1-}
  python3 -c "from savant.utils.welcome import get_starting_message; print(get_starting_message('$name'))"
}
