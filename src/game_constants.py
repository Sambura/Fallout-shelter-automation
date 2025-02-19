import numpy as np

# Rooms detection
undiscovered_room_color = np.array([0, 0, 0])
undiscovered_room_color_max_deviation = 15 # why is that here again?
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

# critical hits
critical_cue_color = np.array([253, 222, 0]) # primary color of critical hit button (before and during activation)
critical_cue_fragment_min_pixels = 25        # for cue fragment detection
critical_hit_color = np.array([18, 244, 21]) # color of critical cue when max multiplier is reached
min_critical_pixels = 75                     # min number of cue/critical pixels to detect a crit
critical_progress_color = np.array([251, 220, 0]) # when crit is in progress, its center part has this color instead of critical_cue_color
crit_max_duration = 10      # in seconds, duration of time during which it is possible to score a crit after its activation

structural_color = np.array([0, 0, 0])
enemy_healthbar_border_color = np.array([255, 25, 25])
enemy_healthbar_color_max_deviation = 8
enemy_healthbar_min_border_pixel_count = 40
enemy_healthbar_outline_color = np.array([89, 9, 8])
enemy_healthbar_detect_blending = 0.8

# Detection of room being clicked
room_click_diff_border_width = 0.09

# walk to new room
walk_wait_min_interval = 0.5
room_reached_min_diff = 0.8

# New room analysis
room_analysis_timeout = 5

# Room scanning
structural_scan_begin_timeout = 1.5

# Structural detection
structural_min_pixels = 1000    # min pixel count in fragment to count as structural
structural_min_size = 100       # width OR height should be at least this large to count as structural 

# Structural room detection
struct_min_room_pixels = 400
struct_min_room_border_fraction = 0.3
struct_min_room_clearance_fraction = 0.65

# Structural room matching
struct_room_match_tolerance = 0.05

# Dialogue detection
primary_dialogue_button_color = np.array([5, 90, 7])
dialogue_button_color_max_deviation = 8
min_dialogue_button_rel_size = np.array([0.4, 0.05])
min_dialogue_button_border_fraction = 0.9
dialogue_button_max_center_deviation_rel = 0.05

corpse_particles_base_color = np.array([255, 234, 153])
corpse_min_fragment_pixels = 300
corpse_min_fragment_size = 35
corpse_color_detection_threshold = 0.5

ui_primary_color = np.array([18, 255, 21])
ui_minimal_pixel_count = 50

character_panel_low_color = np.array([0, 119, 1])
character_panel_min_high_pixels = 50000
character_panel_min_low_pixels = 15000

objective_panel_primary_color = np.array([203, 201, 163])
objective_panel_min_pixels = 200000

# mouse interactions

double_click_activation_radius = 38 # presumably is floor(screen_width / 50)
safe_mouse_click_delay = 0.35 # seconds between clicks to not trigger double click
mouse_click_duration = 0.12 # how long should a mouse button be held during a click
