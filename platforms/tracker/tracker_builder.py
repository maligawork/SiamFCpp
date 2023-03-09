from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

from platforms.core.config import cfg
from platforms.tracker.siamfcpp_tracker import SiamFCppTracker

TRACKS = {
          'SiamFCppTracker': SiamFCppTracker
         } 


def build_tracker(model): 
    return TRACKS[cfg.TRACK.TYPE](model) 
