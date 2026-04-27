# Gradient.msg — single thermal gradient vector at one pixel
# Ref: Wiedemann et al. 2021 DOI:10.1016/j.robot.2020.103687
std_msgs/Header header
float32 pixel_x
float32 pixel_y
float32 grad_x
float32 grad_y
float32 magnitude
float32 direction_rad
float32 temperature_celsius
