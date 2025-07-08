"""
traj_transforms.py

Contains trajectory transforms used in the orca data pipeline. Trajectory transforms operate on a dictionary
that represents a single trajectory, meaning each tensor has the same leading dimension (the trajectory length).
"""

import logging
from typing import Dict

import tensorflow as tf

"""
# 关于轨迹(traj)的使用

不是只使用前9个，而是根据window_size和future_action_window_size参数对整个轨迹进行滑动窗口处理。让我详细解释：

## 轨迹处理机制

从chunk_act_obs函数可以看出：

traj_len = tf.shape(traj["action"])[0]  # 轨迹总长度

effective_traj_len = traj_len - future_action_window_size  # 有效轨迹长度

这意味着：

- 整个轨迹都被使用，但被分成了effective_traj_len个样本

- 每个样本包含：

- window_size个观察值（当前观察值和window_size-1个历史观察值）

- window_size + future_action_window_size个动作（历史动作、当前动作和未来动作）

## 具体示例

假设：

- 原始轨迹长度traj_len = 100

- window_size = 8

- future_action_window_size = 7

那么：

- effective_traj_len = 100 - 7 = 93

- 会生成93个样本

- 第一个样本：

- 观察值：轨迹中的第[0,0,0,0,0,0,0,1]个观察（前7个被填充为第0个，因为没有更早的历史）

- 动作：轨迹中的第[0,0,0,0,0,0,0,1,2,3,4,5,6,7,8]个动作

- 第二个样本：

- 观察值：轨迹中的第[0,0,0,0,0,0,1,2]个观察

- 动作：轨迹中的第[0,0,0,0,0,0,1,2,3,4,5,6,7,8,9]个动作

- 以此类推...

## 关键点

- 滑动窗口：函数使用滑动窗口方式处理整个轨迹

- 填充处理：轨迹开始部分的历史数据不足时，使用第0个时间步的数据进行填充

- 有效长度：轨迹末尾部分由于未来动作不足，不会生成样本，有效长度为traj_len - future_action_window_size

所以，不是只使用前9个时间步，而是使用整个轨迹生成多个重叠的窗口样本，每个样本包含历史、当前和未来的信息。
"""

def chunk_act_obs(traj: Dict, window_size: int, future_action_window_size: int = 0) -> Dict:
    """
    Chunks actions and observations into the given window_size.

    "observation" keys are given a new axis (at index 1) of size `window_size` containing `window_size - 1`
    observations from the past and the current observation. "action" is given a new axis (at index 1) of size
    `window_size + future_action_window_size` containing `window_size - 1` actions from the past, the current
    action, and `future_action_window_size` actions from the future. "pad_mask" is added to "observation" and
    indicates whether an observation should be considered padding (i.e. if it had come from a timestep
    before the start of the trajectory).
    """
    traj_len = tf.shape(traj["action"])[0]
    action_dim = traj["action"].shape[-1]
    effective_traj_len = traj_len - future_action_window_size
    chunk_indices = tf.broadcast_to(tf.range(-window_size + 1, 1), [effective_traj_len, window_size]) + tf.broadcast_to(
        tf.range(effective_traj_len)[:, None], [effective_traj_len, window_size]
    )

    action_chunk_indices = tf.broadcast_to(
        tf.range(-window_size + 1, 1 + future_action_window_size),
        [effective_traj_len, window_size + future_action_window_size],
    ) + tf.broadcast_to(
        tf.range(effective_traj_len)[:, None],
        [effective_traj_len, window_size + future_action_window_size],
    )

    floored_chunk_indices = tf.maximum(chunk_indices, 0)

    goal_timestep = tf.fill([effective_traj_len], traj_len - 1)

    floored_action_chunk_indices = tf.minimum(tf.maximum(action_chunk_indices, 0), goal_timestep[:, None])
    
    # floored_chunk_indices has shape [effective_traj_len, window_size]floored_chunk_indices 有形状 [effective_traj_len, window_size] 
    # When we gather from an array with shape [traj_len, obs_dim] using these indices当我们从形状为 [traj_len， obs_dim] 的数组中收集时，使用这些索引
    # The resulting shape becomes [effective_traj_len, window_size, obs_dim]生成的形状将变为 [effective_traj_len, window_size, obs_dim]
    # tf gather会增加一个dim, 

    #  traj[“observation”] 本身可能是一个嵌套结构（就像张量字典），而不是一个张量。以下是它的作用：
# It takes each individual tensor within the nested traj["observation"] structure它采用嵌套 traj[“observation”] 结构中的每个单独张量
# Applies the tf.gather(x, floored_chunk_indices) operation to that tensor将 tf.gather(x, floored_chunk_indices) 作应用于该张量

    traj["observation"] = tf.nest.map_structure(lambda x: tf.gather(x, floored_chunk_indices), traj["observation"])
    traj["action"] = tf.gather(traj["action"], floored_action_chunk_indices)

    # indicates whether an entire observation is padding
    traj["observation"]["pad_mask"] = chunk_indices >= 0

    # Truncate other elements of the trajectory dict
    traj["task"] = tf.nest.map_structure(lambda x: tf.gather(x, tf.range(effective_traj_len)), traj["task"])
    traj["dataset_name"] = tf.gather(traj["dataset_name"], tf.range(effective_traj_len))
    traj["absolute_action_mask"] = tf.gather(traj["absolute_action_mask"], tf.range(effective_traj_len))

    return traj

def chunk_act_obs_no_padding(traj: Dict, window_size: int, future_action_window_size: int = 0) -> Dict:
    """
    Chunks actions and observations into the given window_size.

    "observation" keys are given a new axis (at index 1) of size `window_size` containing `window_size - 1`
    observations from the past and the current observation. "action" is given a new axis (at index 1) of size
    `window_size + future_action_window_size` containing `window_size - 1` actions from the past, the current
    action, and `future_action_window_size` actions from the future. "pad_mask" is added to "observation" and
    indicates whether an observation should be considered padding (i.e. if it had come from a timestep
    before the start of the trajectory).
    """
    traj_len = tf.shape(traj["action"])[0]
    action_dim = traj["action"].shape[-1]
    effective_traj_len = traj_len - future_action_window_size
    chunk_indices = tf.broadcast_to(tf.range(-window_size + 1, 1), [effective_traj_len, window_size]) + tf.broadcast_to(
        tf.range(effective_traj_len)[:, None], [effective_traj_len, window_size]
    )

    action_chunk_indices = tf.broadcast_to(
        tf.range(-window_size + 1, 1 + future_action_window_size),
        [effective_traj_len, window_size + future_action_window_size],
    ) + tf.broadcast_to(
        tf.range(effective_traj_len)[:, None],
        [effective_traj_len, window_size + future_action_window_size],
    )

    floored_chunk_indices = tf.maximum(chunk_indices, 0)

    goal_timestep = tf.fill([effective_traj_len], traj_len - 1)

    floored_action_chunk_indices = tf.minimum(tf.maximum(action_chunk_indices, 0), goal_timestep[:, None])
    
    # floored_chunk_indices has shape [effective_traj_len, window_size]floored_chunk_indices 有形状 [effective_traj_len, window_size] 
    # When we gather from an array with shape [traj_len, obs_dim] using these indices当我们从形状为 [traj_len， obs_dim] 的数组中收集时，使用这些索引
    # The resulting shape becomes [effective_traj_len, window_size, obs_dim]生成的形状将变为 [effective_traj_len, window_size, obs_dim]
    # tf gather会增加一个dim, 

    #  traj[“observation”] 本身可能是一个嵌套结构（就像张量字典），而不是一个张量。以下是它的作用：
# It takes each individual tensor within the nested traj["observation"] structure它采用嵌套 traj[“observation”] 结构中的每个单独张量
# Applies the tf.gather(x, floored_chunk_indices) operation to that tensor将 tf.gather(x, floored_chunk_indices) 作应用于该张量

    traj["observation"] = tf.nest.map_structure(lambda x: tf.gather(x, floored_chunk_indices), traj["observation"])
    traj["action"] = tf.gather(traj["action"], floored_action_chunk_indices)

    # indicates whether an entire observation is padding
    traj["observation"]["pad_mask"] = chunk_indices >= 0 # 创建了, 但是好像并没有被用到, 我们没有筛选. 假设他没有被用到, 

    # Truncate other elements of the trajectory dict
    traj["task"] = tf.nest.map_structure(lambda x: tf.gather(x, tf.range(effective_traj_len)), traj["task"])
    traj["dataset_name"] = tf.gather(traj["dataset_name"], tf.range(effective_traj_len))
    traj["absolute_action_mask"] = tf.gather(traj["absolute_action_mask"], tf.range(effective_traj_len))


    # Drop the first window_size elements from observation, action, and pad_mask
    traj["observation"] = tf.nest.map_structure(
        lambda x: x[window_size:], traj["observation"]
    )
    # Note: traj["observation"]["pad_mask"] is part of traj["observation"], so it's already sliced above.
    # If it were separate, it would be:
    # traj["observation"]["pad_mask"] = traj["observation"]["pad_mask"][window_size:]

    traj["action"] = traj["action"][window_size:]

    # Adjust other trajectory elements to match the new truncated length
    # They were previously aligned with `effective_traj_len`. Now they need to be aligned with
    # `max(0, effective_traj_len - window_size)`. We achieve this by slicing them from `window_size` as well.
    traj["task"] = tf.nest.map_structure(
        lambda x: x[window_size:], traj["task"]
    )
    traj["dataset_name"] = traj["dataset_name"][window_size:]
    traj["absolute_action_mask"] = traj["absolute_action_mask"][window_size:]

    return traj


def subsample(traj: Dict, subsample_length: int) -> Dict:
    """Subsamples trajectories to the given length."""
    traj_len = tf.shape(traj["action"])[0]
    if traj_len > subsample_length:
        indices = tf.random.shuffle(tf.range(traj_len))[:subsample_length]
        traj = tf.nest.map_structure(lambda x: tf.gather(x, indices), traj)

    return traj


def add_pad_mask_dict(traj: Dict) -> Dict:
    """
    Adds a dictionary indicating which elements of the observation/task should be treated as padding.
        =>> traj["observation"|"task"]["pad_mask_dict"] = {k: traj["observation"|"task"][k] is not padding}
    """
    traj_len = tf.shape(traj["action"])[0]

    for key in ["observation", "task"]:
        pad_mask_dict = {}
        for subkey in traj[key]:
            # Handles "language_instruction", "image_*", and "depth_*"
            if traj[key][subkey].dtype == tf.string:
                pad_mask_dict[subkey] = tf.strings.length(traj[key][subkey]) != 0

            # All other keys should not be treated as padding
            else:
                pad_mask_dict[subkey] = tf.ones([traj_len], dtype=tf.bool)

        traj[key]["pad_mask_dict"] = pad_mask_dict

    return traj
