# ThermalMap.msg — world-frame fused thermal grid map.
# Coordinates are in the frame named by header.frame_id, currently "world".
std_msgs/Header header
uint32 width
uint32 height
float32 resolution
float32 origin_x
float32 origin_y
float32[] temperature_mean
float32[] temperature_variance
float32[] confidence
uint32[] visit_count
float32[] last_seen_age_s
uint8[] view_state
uint8[] view_sectors
string measurement_type
float32[] last_view_distance_m
