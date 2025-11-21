"""Multimodal context fusion for EDON v2."""

import numpy as np
from typing import Dict, Optional, List, Any
from app.v2.schemas_v2 import (
    PhysioInput, MotionInput, EnvInput, VisionInput, 
    AudioInput, TaskInput, SystemInput, CAVRequestV2
)


def extract_physio_features(physio: Optional[PhysioInput]) -> Dict[str, float]:
    """Extract features from physiological signals."""
    if physio is None:
        return {}
    
    features = {}
    
    # EDA features
    if physio.EDA:
        eda_arr = np.array(physio.EDA)
        features['eda_mean'] = float(np.nanmean(eda_arr))
        features['eda_std'] = float(np.nanstd(eda_arr))
        features['eda_max'] = float(np.nanmax(eda_arr))
    
    # BVP features
    if physio.BVP:
        bvp_arr = np.array(physio.BVP)
        features['bvp_mean'] = float(np.nanmean(bvp_arr))
        features['bvp_std'] = float(np.nanstd(bvp_arr))
    
    # Note: Accelerometer moved to MotionInput in v2
    
    # Temperature features
    if physio.TEMP:
        temp_arr = np.array(physio.TEMP)
        features['temp_mean'] = float(np.nanmean(temp_arr))
        features['temp_std'] = float(np.nanstd(temp_arr))
    
    return features


def extract_motion_features(motion: Optional[MotionInput]) -> Dict[str, float]:
    """Extract features from motion/torque data."""
    if motion is None:
        return {}
    
    features = {}
    
    # Accelerometer features (primary motion signal)
    if motion.ACC_x and motion.ACC_y and motion.ACC_z:
        acc_x = np.array(motion.ACC_x)
        acc_y = np.array(motion.ACC_y)
        acc_z = np.array(motion.ACC_z)
        acc_mag = np.sqrt(acc_x**2 + acc_y**2 + acc_z**2)
        features['acc_mean'] = float(np.nanmean(acc_mag))
        features['acc_std'] = float(np.nanstd(acc_mag))
        features['acc_max'] = float(np.nanmax(acc_mag))
    
    # Use pre-computed features if available
    if motion.velocity_magnitude is not None:
        features['velocity_mag'] = motion.velocity_magnitude
    
    if motion.torque_mean is not None:
        features['torque_mean'] = motion.torque_mean
    
    if motion.force_mean is not None:
        features['force_mean'] = motion.force_mean
    
    # Compute from raw signals if available
    if motion.velocity:
        vel_arr = np.array(motion.velocity)
        if len(vel_arr) > 0:
            features['velocity_mag'] = float(np.linalg.norm(vel_arr) if vel_arr.ndim == 1 else np.nanmean(vel_arr))
    
    if motion.torque:
        torque_arr = np.array(motion.torque)
        features['torque_mean'] = float(np.nanmean(torque_arr))
        features['torque_std'] = float(np.nanstd(torque_arr))
        features['torque_max'] = float(np.nanmax(torque_arr))
    
    if motion.force:
        force_arr = np.array(motion.force)
        features['force_mean'] = float(np.nanmean(force_arr))
        features['force_max'] = float(np.nanmax(force_arr))
    
    if motion.acceleration:
        acc_arr = np.array(motion.acceleration)
        features['accel_mag'] = float(np.linalg.norm(acc_arr) if acc_arr.ndim == 1 else np.nanmean(acc_arr))
    
    return features


def extract_env_features(env: Optional[EnvInput]) -> Dict[str, float]:
    """Extract features from environmental context."""
    if env is None:
        return {}
    
    features = {}
    
    if env.temp_c is not None:
        features['temp_c'] = env.temp_c
    
    if env.humidity is not None:
        features['humidity'] = env.humidity
    
    if env.aqi is not None:
        features['aqi'] = float(env.aqi)
    
    if env.local_hour is not None:
        features['local_hour'] = float(env.local_hour)
        # Circular encoding for hour
        features['hour_sin'] = np.sin(2 * np.pi * env.local_hour / 24.0)
        features['hour_cos'] = np.cos(2 * np.pi * env.local_hour / 24.0)
    
    if env.pressure is not None:
        features['pressure'] = env.pressure
    
    if env.light_level is not None:
        features['light_level'] = env.light_level
    
    if env.noise_level is not None:
        features['noise_level'] = env.noise_level
    
    return features


def extract_vision_features(vision: Optional[VisionInput]) -> Dict[str, Any]:
    """Extract features from vision context."""
    if vision is None:
        return {}
    
    features = {}
    
    if vision.embedding:
        # Use embedding directly (normalized)
        emb = np.array(vision.embedding)
        if len(emb) > 0:
            # Normalize embedding
            emb_norm = emb / (np.linalg.norm(emb) + 1e-8)
            features['vision_embedding'] = emb_norm.tolist()
            features['vision_embedding_norm'] = float(np.linalg.norm(emb))
    
    if vision.objects:
        features['num_objects'] = len(vision.objects)
        features['has_objects'] = 1.0 if len(vision.objects) > 0 else 0.0
    
    if vision.scene_type:
        # One-hot encode common scene types
        scene_map = {'indoor': 0, 'outdoor': 1, 'vehicle': 2, 'unknown': 3}
        features['scene_type'] = scene_map.get(vision.scene_type.lower(), 3)
    
    if vision.activity_context:
        # One-hot encode common activities
        activity_map = {'walking': 0, 'sitting': 1, 'operating': 2, 'resting': 3, 'unknown': 4}
        features['activity_context'] = activity_map.get(vision.activity_context.lower(), 4)
    
    return features


def extract_audio_features(audio: Optional[AudioInput]) -> Dict[str, Any]:
    """Extract features from audio context."""
    if audio is None:
        return {}
    
    features = {}
    
    if audio.embedding:
        # Use embedding directly (normalized)
        emb = np.array(audio.embedding)
        if len(emb) > 0:
            # Normalize embedding
            emb_norm = emb / (np.linalg.norm(emb) + 1e-8)
            features['audio_embedding'] = emb_norm.tolist()
            features['audio_embedding_norm'] = float(np.linalg.norm(emb))
    
    if audio.keywords:
        features['num_keywords'] = len(audio.keywords)
        features['has_keywords'] = 1.0 if len(audio.keywords) > 0 else 0.0
        # Check for stress-related keywords
        stress_keywords = ['help', 'stop', 'danger', 'emergency', 'error', 'fail']
        has_stress_kw = any(kw.lower() in stress_keywords for kw in audio.keywords)
        features['has_stress_keywords'] = 1.0 if has_stress_kw else 0.0
    
    if audio.speech_activity is not None:
        features['speech_activity'] = audio.speech_activity
    
    if audio.emotion:
        # One-hot encode emotions
        emotion_map = {'calm': 0, 'stressed': 1, 'excited': 2, 'neutral': 3, 'unknown': 4}
        features['emotion'] = emotion_map.get(audio.emotion.lower(), 4)
    
    return features


def extract_task_features(task: Optional[TaskInput]) -> Dict[str, float]:
    """Extract features from task metadata."""
    if task is None:
        return {}
    
    features = {}
    
    if task.confidence is not None:
        features['task_confidence'] = task.confidence
    
    if task.priority is not None:
        features['task_priority'] = float(task.priority) / 10.0  # Normalize to [0-1]
    
    if task.complexity is not None:
        features['task_complexity'] = task.complexity
    
    if task.deadline_proximity is not None:
        features['deadline_proximity'] = task.deadline_proximity
    
    # Combined task stress indicator
    if task.complexity is not None and task.deadline_proximity is not None:
        features['task_stress'] = (task.complexity + task.deadline_proximity) / 2.0
    
    return features


def extract_system_features(system: Optional[SystemInput]) -> Dict[str, float]:
    """Extract features from system/robotics signals."""
    if system is None:
        return {}
    
    features = {}
    
    if system.cpu_usage is not None:
        features['cpu_usage'] = system.cpu_usage
    
    if system.memory_usage is not None:
        features['memory_usage'] = system.memory_usage
    
    if system.network_latency is not None:
        features['network_latency'] = system.network_latency / 1000.0  # Normalize (assuming ms)
    
    if system.error_rate is not None:
        features['error_rate'] = system.error_rate
    
    if system.battery_level is not None:
        features['battery_level'] = system.battery_level
    
    if system.system_load is not None:
        features['system_load'] = system.system_load
    
    # Combined system stress indicator
    stress_indicators = []
    if system.cpu_usage is not None:
        stress_indicators.append(system.cpu_usage)
    if system.memory_usage is not None:
        stress_indicators.append(system.memory_usage)
    if system.error_rate is not None:
        stress_indicators.append(system.error_rate)
    
    if stress_indicators:
        features['system_stress'] = float(np.mean(stress_indicators))
    
    return features


def fuse_multimodal_features(request: CAVRequestV2) -> Dict[str, Any]:
    """
    Fuse all multimodal inputs into a unified feature representation.
    
    Returns:
        Dictionary with:
        - 'features': Dict of scalar features
        - 'embeddings': Dict of embedding vectors
        - 'modalities_present': List of present modalities
    """
    all_features = {}
    embeddings = {}
    modalities_present = []
    
    # Extract features from each modality
    if request.physio:
        physio_feat = extract_physio_features(request.physio)
        all_features.update(physio_feat)
        modalities_present.append('physio')
    
    if request.motion:
        motion_feat = extract_motion_features(request.motion)
        all_features.update(motion_feat)
        modalities_present.append('motion')
    
    if request.env:
        env_feat = extract_env_features(request.env)
        all_features.update(env_feat)
        modalities_present.append('env')
    
    if request.vision:
        vision_feat = extract_vision_features(request.vision)
        # Separate embeddings from scalar features
        if 'vision_embedding' in vision_feat:
            embeddings['vision'] = vision_feat.pop('vision_embedding')
        all_features.update(vision_feat)
        modalities_present.append('vision')
    
    if request.audio:
        audio_feat = extract_audio_features(request.audio)
        # Separate embeddings from scalar features
        if 'audio_embedding' in audio_feat:
            embeddings['audio'] = audio_feat.pop('audio_embedding')
        all_features.update(audio_feat)
        modalities_present.append('audio')
    
    if request.task:
        task_feat = extract_task_features(request.task)
        all_features.update(task_feat)
        modalities_present.append('task')
    
    if request.system:
        system_feat = extract_system_features(request.system)
        all_features.update(system_feat)
        modalities_present.append('system')
    
    return {
        'features': all_features,
        'embeddings': embeddings,
        'modalities_present': modalities_present
    }

