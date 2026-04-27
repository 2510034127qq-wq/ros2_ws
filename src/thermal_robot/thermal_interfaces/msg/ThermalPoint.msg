# ThermalPoint.msg — single detected hotspot in 2D image space
# Ref: Wiedemann et al. 2021 DOI:10.1016/j.robot.2020.103687
std_msgs/Header header
float32 pixel_x
float32 pixel_y
geometry_msgs/Point world_position
float32 temperature_celsius
float32 intensity
float32 confidence
