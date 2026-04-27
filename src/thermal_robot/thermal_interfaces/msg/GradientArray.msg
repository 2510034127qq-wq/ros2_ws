# GradientArray.msg — array of thermal gradient vectors
# Published on /thermal/gradient at 10 Hz
# Ref: Wiedemann et al. 2021 DOI:10.1016/j.robot.2020.103687
std_msgs/Header header
thermal_interfaces/Gradient[] gradients
float32 max_magnitude
uint32  peak_pixel_x
uint32  peak_pixel_y
float32 peak_temperature_celsius
string  method
