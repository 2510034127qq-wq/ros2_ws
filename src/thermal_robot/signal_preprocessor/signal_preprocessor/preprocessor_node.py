#!/usr/bin/env python3
"""
preprocessor_node.py — Signal Preprocessor Node v2
====================================================
v2 优化（基于 20260315 仿真测试数据）：
  问题：Preproc Effect=0.16（接近透明滤波），raw/filtered 图像几乎无差异。
  根因：Kalman R=0.25 远小于 Q=0.01，R/Q=25，滤波器过度相信每帧观测值，
        等效于无滤波（K ≈ R/(R+Q) 趋近 1，输出≈输入）。
  修复1：kalman_r: 0.25 → 4.0，kalman_q: 0.01 → 0.08。
         R/Q=50，K≈0.08，即每帧仅修正 8% 的预测偏差，有效抑制 0.5°C 噪声。
  修复2：新增空间高斯平滑（3×3 窗口，sigma=0.8px）。
         纯时间滤波保留了空间噪声（相邻像素跳变），空间滤波后梯度计算更稳定，
         梯度幅值信噪比预计提升 2–3 倍。

Sub  : /sim/thermal_raw   sensor_msgs/Image 32FC1
Pub  : /thermal/filtered  sensor_msgs/Image 32FC1
filter_method param: "kalman" | "moving_avg" | "exp_smooth"

Kalman 1-D per-pixel：
  Predict: x_hat = x_prev;  P_ = P + Q
  Update:  K = P_/(P_+R);  x = x_hat + K*(z-x_hat);  P = (1-K)*P_
  v2 默认：Q=0.08, R=4.0 → K≈0.02（低卡尔曼增益 = 强平滑）

Ref: Nakagawa 2020 DOI:10.1109/JSEN.2020.2984234
     空间平滑：Reggente 2009 DOI:10.1109/ICSENS.2009.5398427
"""

from collections import deque
import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import (QoSProfile, QoSReliabilityPolicy,
                        QoSHistoryPolicy, QoSDurabilityPolicy)
from sensor_msgs.msg import Image


def _gaussian_kernel(size: int, sigma: float) -> np.ndarray:
    """生成归一化 2-D 高斯卷积核。size 必须为奇数。"""
    k = size // 2
    y, x = np.mgrid[-k:k+1, -k:k+1]
    kernel = np.exp(-(x**2 + y**2) / (2.0 * sigma**2))
    return (kernel / kernel.sum()).astype(np.float32)


def _conv2d_valid(arr: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    """
    纯 NumPy 2-D 卷积（边界 edge-pad，无需 scipy）。
    arr: (H, W) float32；kernel: (k, k) float32，k 为奇数。
    """
    k = kernel.shape[0] // 2
    padded = np.pad(arr, k, mode='edge')
    H, W   = arr.shape
    out    = np.zeros((H, W), dtype=np.float32)
    for i in range(kernel.shape[0]):
        for j in range(kernel.shape[1]):
            out += kernel[i, j] * padded[i:i+H, j:j+W]
    return out


class PreprocessorNode(Node):
    """
    热图像时序 + 空间滤波节点。

    处理流程：
      原始帧 z  →  [时序滤波 Kalman/EMA/MovAvg]  →  [空间高斯平滑]  →  输出

    空间平滑可通过 spatial_smooth=false 单独禁用，用于对比实验。
    """

    def __init__(self):
        super().__init__('preprocessor_node')

        # ── 参数声明 ─────────────────────────────────────────────────────────
        self.declare_parameter('filter_method',     'kalman')
        self.declare_parameter('frame_id',          'thermal_camera')
        self.declare_parameter('moving_avg_window', 8)
        self.declare_parameter('exp_smooth_alpha',  0.2)
        # v2 修正默认值：强化 Kalman 平滑
        self.declare_parameter('kalman_q',          0.08)   # v2: 0.01→0.08
        self.declare_parameter('kalman_r',          4.0)    # v2: 0.25→4.0
        # v2 新增：空间平滑
        self.declare_parameter('spatial_smooth',    True)
        self.declare_parameter('spatial_sigma',     0.8)    # 单位：像素
        self.declare_parameter('spatial_kernel_size', 3)    # 奇数

        # ── 读取参数 ─────────────────────────────────────────────────────────
        self._method  = self.get_parameter('filter_method').value
        self._frame   = self.get_parameter('frame_id').value
        self._win     = int(self.get_parameter('moving_avg_window').value)
        self._alpha   = float(self.get_parameter('exp_smooth_alpha').value)
        self._Q       = float(self.get_parameter('kalman_q').value)
        self._R       = float(self.get_parameter('kalman_r').value)
        self._spatial = bool(self.get_parameter('spatial_smooth').value)
        self._sp_sig  = float(self.get_parameter('spatial_sigma').value)
        self._sp_ksz  = int(self.get_parameter('spatial_kernel_size').value)
        # kernel size 强制奇数
        if self._sp_ksz % 2 == 0:
            self._sp_ksz += 1

        # ── 预计算空间卷积核 ─────────────────────────────────────────────────
        if self._spatial:
            self._sp_kernel = _gaussian_kernel(self._sp_ksz, self._sp_sig)
        else:
            self._sp_kernel = None

        # ── 滤波状态 ─────────────────────────────────────────────────────────
        self._buf  : deque             = deque(maxlen=self._win)
        self._ema  : np.ndarray | None = None
        self._kx   : np.ndarray | None = None
        self._kP   : np.ndarray | None = None
        self._frame_count = 0

        # ── 统计（用于日志验证滤波效果）────────────────────────────────────
        self._snr_log_interval = 100   # 每 100 帧输出一次信噪比对比

        # ── QoS ─────────────────────────────────────────────────────────────
        qos = QoSProfile(
            reliability=QoSReliabilityPolicy.BEST_EFFORT,
            history=QoSHistoryPolicy.KEEP_LAST, depth=5,
            durability=QoSDurabilityPolicy.VOLATILE)
        self._sub = self.create_subscription(
            Image, '/sim/thermal_raw', self._cb, qos)
        self._pub = self.create_publisher(
            Image, '/thermal/filtered', qos)

        # ── 原始帧缓存（用于滤波效果 SNR 对比）──────────────────────────────
        self._raw_std_buf  : deque = deque(maxlen=50)
        self._filt_std_buf : deque = deque(maxlen=50)

        K_approx = self._Q / (self._Q + self._R) if (self._Q + self._R) > 0 else 0
        self.get_logger().info(
            f'preprocessor_node v2 | method={self._method} '
            f'| Q={self._Q} R={self._R} K≈{K_approx:.3f} '
            f'| spatial_smooth={self._spatial} sigma={self._sp_sig}px '
            f'| moving_avg_win={self._win}')

    # ── 图像辅助 ─────────────────────────────────────────────────────────────

    def _arr(self, msg: Image) -> np.ndarray:
        n = msg.width * msg.height
        return np.frombuffer(bytes(msg.data[:n * 4]),
                             np.float32).reshape(msg.height, msg.width).copy()

    def _pack(self, arr: np.ndarray, msg: Image) -> Image:
        out             = Image()
        out.header      = msg.header
        out.header.frame_id = self._frame
        out.height      = arr.shape[0]
        out.width       = arr.shape[1]
        out.encoding    = '32FC1'
        out.is_bigendian = False
        out.step        = arr.shape[1] * 4
        out.data        = arr.astype(np.float32).tobytes()
        return out

    # ── 时序滤波 ─────────────────────────────────────────────────────────────

    def _temporal_filter(self, z: np.ndarray) -> np.ndarray:
        """对每帧应用选定的时序滤波方法。"""
        if self._method == 'moving_avg':
            self._buf.append(z.copy())
            return np.mean(np.stack(self._buf), axis=0)

        elif self._method == 'exp_smooth':
            self._ema = self._alpha * z + (1.0 - self._alpha) * self._ema
            return self._ema.copy()

        else:  # kalman（默认）
            # Predict
            Pp = self._kP + self._Q
            # Update
            K        = Pp / (Pp + self._R)
            self._kx = self._kx + K * (z - self._kx)
            self._kP = (1.0 - K) * Pp
            return self._kx.copy()

    # ── 空间平滑 ─────────────────────────────────────────────────────────────

    def _spatial_smooth(self, arr: np.ndarray) -> np.ndarray:
        """3×3 高斯空间平滑，使用预计算核卷积。"""
        if self._sp_kernel is None:
            return arr
        return _conv2d_valid(arr, self._sp_kernel)

    # ── 主回调 ───────────────────────────────────────────────────────────────

    def _cb(self, msg: Image):
        z = self._arr(msg)
        self._frame_count += 1

        # 首帧初始化
        if self._kx is None:
            self._kx  = z.copy()
            self._kP  = np.ones_like(z)
            self._ema = z.copy()
            for _ in range(self._win):
                self._buf.append(z.copy())

        # 记录原始噪声水平
        self._raw_std_buf.append(float(z.std()))

        # 时序滤波
        temporal_out = self._temporal_filter(z)

        # 空间高斯平滑（可选）
        if self._spatial and self._sp_kernel is not None:
            out = self._spatial_smooth(temporal_out.astype(np.float32))
        else:
            out = temporal_out

        out = out.astype(np.float32)

        # 记录输出噪声水平
        self._filt_std_buf.append(float(out.std()))

        self._pub.publish(self._pack(out, msg))

        # 定期日志：显示噪声抑制效果
        if self._frame_count % self._snr_log_interval == 0:
            if len(self._raw_std_buf) >= 10 and len(self._filt_std_buf) >= 10:
                raw_s  = float(np.mean(list(self._raw_std_buf)[-20:]))
                filt_s = float(np.mean(list(self._filt_std_buf)[-20:]))
                ratio  = raw_s / max(filt_s, 1e-6)
                effect = 1.0 - min(1.0, filt_s / max(raw_s, 1e-6))
                self.get_logger().info(
                    f'[FILTER_STATS] frame={self._frame_count} '
                    f'raw_std={raw_s:.3f}°C '
                    f'filt_std={filt_s:.3f}°C '
                    f'SNR_ratio={ratio:.2f}x '
                    f'preproc_effect={effect:.2f}')


def main(args=None):
    rclpy.init(args=args)
    node = PreprocessorNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        try:
            rclpy.shutdown()
        except Exception:
            pass


if __name__ == '__main__':
    main()
