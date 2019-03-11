"""
    This module contains functions that help in sending
    outputs on data configuration and model progress
    to the command line.
"""

import progressbar

progress_bar_widgets = [
    'Training: ', progressbar.Percentage(),
    ' ',
    progressbar.Bar(marker='~',
                    left='[',
                    right=']'
                    ),
    ' ',
    progressbar.ETA(), ' '
]
