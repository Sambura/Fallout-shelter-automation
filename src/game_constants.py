import numpy as np

undiscoveredRoomColor = np.array([0, 0, 0])
undiscovered_room_color_max_deviation = 15
undiscovered_room_border_mean_color = np.array([0, 220, 0])
undiscovered_room_border_color_max_deviation = 40
min_room_match = 0.7
min_room_size = (50, 50)
healing_color_low = np.array([0, 90, 1])
healing_color_high = np.array([1, 255, 2])
healing_button_min_pixels = 200
stimpak_color_ratio = 0.9
stimpak_min_clean_colors = 0.75
antirad_color_ratio = 0.95
antirad_min_clean_colors = 0.8
levelup_color_ratio = 0
levelup_min_clean_colors = 0.8
color_ratio_tolerance = 0.15
critical_cue_color = np.array([253, 222, 0])
critical_cue_fragment_min_pixels = 25
critical_hit_color = np.array([17, 243, 20])
critical_hit_color_deviation = 10