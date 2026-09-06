# SourceEstimate.msg — candidate or confirmed thermal source estimate.
std_msgs/Header header
string id
string status
geometry_msgs/Point position
float32 covariance_xx
float32 covariance_xy
float32 covariance_yy
float32 strength
float32 sigma
float32 existence_probability
float32 confidence
uint32 observations
float32 age_s
