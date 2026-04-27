# ThermalField.msg — full reconstructed thermal scalar field
# Published on /thermal/field at 10 Hz
# Ref: Reggente & Lilienthal 2009 DOI:10.1109/ICSENS.2009.5398427
std_msgs/Header header
uint32  width
uint32  height
float32 resolution_x
float32 resolution_y
float32 origin_x
float32 origin_y
float32[] data
float32 min_temperature_celsius
float32 max_temperature_celsius
float32 mean_temperature_celsius
float32 std_temperature_celsius
thermal_interfaces/ThermalPoint[] hotspots
string reconstruction_method
string source_encoding
