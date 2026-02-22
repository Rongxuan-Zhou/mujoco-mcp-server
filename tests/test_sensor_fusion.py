import pytest
import numpy as np
from mujoco_mcp.sensor_feedback import SensorFusion, SensorType, SensorReading, LowPassFilter
import time

def test_low_pass_filter():
    """LowPassFilter must accept n_channels and filter signals."""
    lpf = LowPassFilter(cutoff_freq=10.0, n_channels=3, dt=0.01)
    signal = np.array([1.0, 2.0, 3.0])
    filtered = lpf.update(signal)
    assert filtered.shape == signal.shape

def test_sensor_fusion_empty():
    """SensorFusion with no valid readings returns empty dict."""
    fusion = SensorFusion()
    result = fusion.fuse_sensor_data([])
    assert result == {}

def test_sensor_reading_creation():
    """SensorReading must be created with correct fields."""
    reading = SensorReading(
        sensor_id="joint_pos_0",
        sensor_type=SensorType.JOINT_POSITION,
        data=np.array([0.1, 0.2, 0.3]),
        timestamp=time.time(),
        quality=0.9,
    )
    assert reading.quality == 0.9
    assert reading.data.shape == (3,)

def test_sensor_fusion_single():
    """SensorFusion with one valid reading returns that reading's data."""
    fusion = SensorFusion()
    fusion.add_sensor("joint_pos_0", SensorType.JOINT_POSITION, weight=1.0)
    reading = SensorReading(
        sensor_id="joint_pos_0",
        sensor_type=SensorType.JOINT_POSITION,
        data=np.array([1.0, 2.0]),
        timestamp=time.time(),
        quality=1.0,
    )
    result = fusion.fuse_sensor_data([reading])
    assert SensorType.JOINT_POSITION.value in result
    np.testing.assert_allclose(result[SensorType.JOINT_POSITION.value], [1.0, 2.0], atol=1e-6)
