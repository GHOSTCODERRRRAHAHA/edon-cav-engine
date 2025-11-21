"""EDON v2 Engine - Multimodal context fusion with PCA and neural head."""

import numpy as np
import math
from typing import Dict, Any, List, Optional, Tuple, Set
from app.v2.schemas_v2 import CAVRequestV2, CAVResponseV2, InfluenceFields
from app.v2.multimodal_fusion import fuse_multimodal_features
from app.v2.state_classifier_v2 import classify_state_v2, compute_influence_fields
from app.v2.device_profiles import (
    DeviceProfile, DeviceProfileConfig, get_profile,
    get_available_modalities
)
from app.v2.pca_fusion import PCAFusion, create_default_pca_fusion
from app.v2.neural_head import NeuralHeadMLP, create_default_neural_head


class CAVEngineV2:
    """EDON v2 CAV Engine with multimodal fusion, PCA, and neural head."""
    
    def __init__(self, device_profile: Optional[str] = None):
        """
        Initialize v2 engine.
        
        Args:
            device_profile: Device profile name (humanoid_full, wearable_limited, drone_nav)
        """
        import os
        import logging
        logger = logging.getLogger(__name__)
        
        # Device profile - only set if explicitly provided
        self.device_profile: Optional[DeviceProfileConfig] = None
        if device_profile:
            profile_obj = get_profile(device_profile)
            if profile_obj:
                self.device_profile = profile_obj
                self.weights = profile_obj.modality_weights.copy()
            else:
                # Invalid profile name - use default weights
                self.weights = {
                    'physio': 0.4,
                    'motion': 0.15,
                    'env': 0.15,
                    'vision': 0.1,
                    'audio': 0.1,
                    'task': 0.05,
                    'system': 0.05
                }
        else:
            # No profile - use default weights
            self.weights = {
                'physio': 0.4,
                'motion': 0.15,
                'env': 0.15,
                'vision': 0.1,
                'audio': 0.1,
                'task': 0.05,
                'system': 0.05
            }
        
        # PCA fusion for 128-dim embeddings
        self.pca_fusion = create_default_pca_fusion()
        self.pca_fitted = False
        
        # Try to load PCA from environment (handled in main.py, skip here to avoid duplicate loading)
        # PCA will be loaded in main.py if needed
        
        # Neural head for state prediction
        self.neural_head = create_default_neural_head(input_dim=128)
        
        # Neural weights loading handled in main.py to avoid duplicate loading
        
        # EMA smoothing for CAV vector
        self.cav_smooth = None
        self.alpha = 0.3  # EMA smoothing factor
        
        # Store recent feature vectors for PCA fitting
        self.recent_features: List[Dict[str, float]] = []
        self.max_recent_features = 100
    
    def compute_cav_v2(self, request: CAVRequestV2, device_profile: Optional[str] = None) -> Dict[str, Any]:
        """
        Compute CAV v2 from multimodal inputs with PCA fusion and neural head.
        
        Args:
            request: CAV request with multimodal inputs
            device_profile: Optional device profile override
            
        Returns:
            Dictionary with:
            - cav_vector: List[float] - 128-dimensional CAV embedding
            - state_class: str - State classification
            - p_stress: float - Probability of stress
            - p_chaos: float - Probability of chaos
            - influences: Dict - Control influence fields
            - confidence: float - Overall confidence
            - metadata: Dict - Additional metadata
        """
        # Handle device profile (OEM-friendly: weighting only, no validation)
        # Use profile from request first, then from parameter, then engine default
        profile = None
        profile_name = None
        
        # Check if profile is explicitly in request
        if hasattr(request, 'device_profile'):
            request_profile_value = getattr(request, 'device_profile', None)
            if request_profile_value is not None and request_profile_value != "":
                profile = get_profile(request_profile_value)
                if profile:
                    profile_name = profile.name
        
        # Use explicit device_profile parameter if provided
        if not profile and device_profile:
            profile = get_profile(device_profile)
            if profile:
                profile_name = profile.name
        
        # Use engine default if no explicit profile
        if not profile and self.device_profile:
            profile = self.device_profile
            profile_name = profile.name
        
        # Apply profile weights (if available) - profile is a hint, not a contract
        # Never reject requests based on missing modalities
        if profile:
            self.weights = profile.modality_weights.copy()
        
        # Fuse multimodal features
        fused = fuse_multimodal_features(request)
        features = fused['features']
        embeddings = fused['embeddings']
        modalities_present = fused['modalities_present']
        
        # Update weights based on profile if available
        if profile:
            # Normalize weights to sum to 1.0
            total_weight = sum(self.weights.get(m, 0.0) for m in modalities_present)
            if total_weight > 0:
                for m in modalities_present:
                    if m in self.weights:
                        self.weights[m] = self.weights[m] / total_weight
        
        # Store features for PCA fitting
        self.recent_features.append(features.copy())
        if len(self.recent_features) > self.max_recent_features:
            self.recent_features.pop(0)
        
        # Fit PCA if not fitted and we have enough samples
        if not self.pca_fitted and len(self.recent_features) >= 2:
            try:
                self.pca_fusion.fit(self.recent_features)
                self.pca_fitted = True
            except Exception as e:
                # PCA fitting failed, will use fallback
                pass
        
        # Compute base scores from each modality (weighted by profile)
        scores = self._compute_modality_scores(features, embeddings, modalities_present)
        
        # Compute probabilities
        p_stress = self._compute_p_stress(scores, features)
        p_focus = self._compute_p_focus(scores, features)
        p_chaos = self._compute_p_chaos(scores, features)
        
        # Compute environmental and circadian scores
        env_score = self._compute_env_score(features)
        circadian_score = self._compute_circadian_score(features)
        system_stress = features.get('system_stress', 0.0)
        
        # Check for emergency indicators
        emergency_indicators = self._check_emergency_indicators(features, request)
        
        # Generate 128-dim CAV embedding using PCA
        try:
            if self.pca_fitted:
                cav_embedding_128 = self.pca_fusion.transform(features)
            else:
                # Fallback: create embedding from features directly
                cav_embedding_128 = self._create_fallback_embedding(features, scores)
        except Exception:
            # Fallback if PCA transform fails
            cav_embedding_128 = self._create_fallback_embedding(features, scores)
        
        # Apply EMA smoothing to embedding
        if self.cav_smooth is None:
            self.cav_smooth = cav_embedding_128.copy()
        else:
            # Ensure same dimension
            if len(self.cav_smooth) == len(cav_embedding_128):
                self.cav_smooth = self.alpha * cav_embedding_128 + (1 - self.alpha) * self.cav_smooth
            else:
                self.cav_smooth = cav_embedding_128.copy()
        
        cav_vector_smooth = self.cav_smooth.tolist()
        
        # Use neural head for state prediction and action recommendations
        neural_pred = self.neural_head.predict(cav_vector_smooth)
        
        # Combine neural head predictions with rule-based classification
        # Use neural head state if confidence is high, otherwise use rule-based
        neural_confidence = neural_pred['confidence']
        rule_based_state = classify_state_v2(
            p_stress=p_stress,
            p_focus=p_focus,
            p_chaos=p_chaos,
            env_score=env_score,
            circadian_score=circadian_score,
            system_stress=system_stress,
            emergency_indicators=emergency_indicators
        )
        
        # Use rule-based classification for state_class (deterministic, demo-friendly)
        # Neural head predictions are included in metadata for reference only
        state_class = rule_based_state
        
        # Use rule-based influences but blend with neural recommendations
        rule_influences = compute_influence_fields(
            state=state_class,
            p_stress=p_stress,
            p_focus=p_focus,
            p_chaos=p_chaos,
            env_score=env_score,
            system_stress=system_stress
        )
        neural_influences = neural_pred['action_recommendations']
        # Blend: 70% rule-based, 30% neural
        influences_dict = {
            'speed_scale': rule_influences['speed_scale'] * 0.7 + neural_influences['speed_scale'] * 0.3,
            'torque_scale': rule_influences['torque_scale'] * 0.7 + neural_influences['torque_scale'] * 0.3,
            'safety_scale': rule_influences['safety_scale'] * 0.7 + neural_influences['safety_scale'] * 0.3,
            'caution_flag': rule_influences['caution_flag'] or neural_influences['caution_flag'],
            'emergency_flag': rule_influences['emergency_flag'] or neural_influences['emergency_flag'],
            'focus_boost': rule_influences['focus_boost'] * 0.7 + neural_influences['focus_boost'] * 0.3,
            'recovery_recommended': rule_influences['recovery_recommended'] or neural_influences['recovery_recommended']
        }
        
        # Compute confidence based on modalities present, feature quality, and profile
        confidence = self._compute_confidence(modalities_present, features)
        if profile:
            confidence += profile.default_confidence_boost
        confidence = min(1.0, confidence)
        
        # Build metadata (matching OEM spec exactly)
        metadata = {
            'modalities_present': modalities_present,
            'num_features': len(features),
            'has_embeddings': len(embeddings) > 0,
            'scores': {
                'physio': float(scores.get('physio', 0.5)),
                'motion': float(scores.get('motion', 0.5)),
                'env': float(scores.get('env', 0.5)),
                'vision': float(scores.get('vision', 0.5)),
                'audio': float(scores.get('audio', 0.5)),
                'task': float(scores.get('task', 0.5)),
                'system': float(scores.get('system', 0.5))
            },
            'device_profile': profile_name,  # None if no profile
            'pca_fitted': self.pca_fitted,
            'neural_confidence': float(neural_confidence),
            'neural_state_probs': neural_pred.get('state_probs', {
                'restorative': 0.0,
                'focus': 0.0,
                'balanced': 0.0,
                'alert': 0.0,
                'overload': 0.0,
                'emergency': 0.0
            })
        }
        
        return {
            'cav_vector': cav_vector_smooth,  # 128-dim embedding
            'state_class': state_class,
            'p_stress': float(p_stress),
            'p_chaos': float(p_chaos),  # Note: p_focus not in v2 spec, only in metadata if needed
            'influences': influences_dict,
            'confidence': float(confidence),
            'metadata': metadata
        }
    
    def _create_fallback_embedding(self, features: Dict[str, float], scores: Dict[str, float]) -> np.ndarray:
        """Create 128-dim embedding from features when PCA is not available."""
        # Combine features and scores into a vector
        feature_values = list(features.values())
        score_values = list(scores.values())
        
        combined = feature_values + score_values
        
        # Pad or truncate to 128 dimensions
        if len(combined) < 128:
            # Pad with zeros
            combined = combined + [0.0] * (128 - len(combined))
        elif len(combined) > 128:
            # Truncate
            combined = combined[:128]
        
        embedding = np.array(combined, dtype=np.float32)
        
        # L2 normalize
        norm = np.linalg.norm(embedding)
        if norm > 0:
            embedding = embedding / norm
        
        return embedding
    
    def _compute_modality_scores(
        self, 
        features: Dict[str, Any], 
        embeddings: Dict[str, List[float]],
        modalities_present: List[str]
    ) -> Dict[str, float]:
        """Compute scores for each modality."""
        scores = {}
        
        # Physiological score
        if 'physio' in modalities_present:
            physio_score = 1.0
            if 'eda_mean' in features:
                # Higher EDA = more stress
                eda_norm = min(1.0, features['eda_mean'] / 2.0)  # Normalize
                physio_score = 1.0 - eda_norm * 0.5
            if 'bvp_mean' in features:
                bvp_norm = min(1.0, features['bvp_mean'] / 1.0)  # Normalize
                physio_score = min(physio_score, 1.0 - bvp_norm * 0.3)
            scores['physio'] = max(0.0, physio_score)
        else:
            scores['physio'] = 0.5  # Neutral if missing
        
        # Motion score
        if 'motion' in modalities_present:
            motion_score = 1.0
            if 'torque_mean' in features:
                # High torque = potential stress
                torque_norm = min(1.0, features['torque_mean'] / 100.0)  # Normalize
                motion_score = 1.0 - torque_norm * 0.4
            if 'velocity_mag' in features:
                # Very high velocity = potential chaos
                vel_norm = min(1.0, features['velocity_mag'] / 10.0)
                motion_score = min(motion_score, 1.0 - vel_norm * 0.2)
            # Use accelerometer std as chaos indicator (high std = chaotic motion)
            if 'acc_std' in features:
                acc_std = features['acc_std']
                # High std indicates chaotic motion
                acc_std_norm = min(1.0, acc_std / 10.0)  # Normalize (std of 10 = very chaotic)
                motion_score = min(motion_score, 1.0 - acc_std_norm * 0.5)
            scores['motion'] = max(0.0, motion_score)
        else:
            scores['motion'] = 0.5
        
        # Environment score
        if 'env' in modalities_present:
            env_score = self._compute_env_score(features)
            scores['env'] = env_score
        else:
            scores['env'] = 0.5
        
        # Vision score (from embeddings or scene context)
        if 'vision' in modalities_present:
            vision_score = 1.0
            if 'vision_embedding_norm' in features:
                # Use embedding norm as indicator (normalized)
                emb_norm = min(1.0, features['vision_embedding_norm'] / 10.0)
                vision_score = 1.0 - emb_norm * 0.3
            scores['vision'] = max(0.0, vision_score)
        else:
            scores['vision'] = 0.5
        
        # Audio score
        if 'audio' in modalities_present:
            audio_score = 1.0
            if 'has_stress_keywords' in features and features['has_stress_keywords'] > 0.5:
                audio_score = 0.3  # Stress keywords indicate problems
            if 'emotion' in features:
                if features['emotion'] == 1:  # stressed
                    audio_score = 0.4
                elif features['emotion'] == 0:  # calm
                    audio_score = 0.9
            scores['audio'] = max(0.0, audio_score)
        else:
            scores['audio'] = 0.5
        
        # Task score
        if 'task' in modalities_present:
            task_score = 1.0
            if 'task_stress' in features:
                task_score = 1.0 - features['task_stress']
            elif 'task_complexity' in features:
                task_score = 1.0 - features['task_complexity'] * 0.5
            scores['task'] = max(0.0, task_score)
        else:
            scores['task'] = 0.5
        
        # System score
        if 'system' in modalities_present:
            system_score = 1.0
            if 'system_stress' in features:
                system_score = 1.0 - features['system_stress']
            scores['system'] = max(0.0, system_score)
        else:
            scores['system'] = 0.5
        
        return scores
    
    def _compute_p_stress(self, scores: Dict[str, float], features: Dict[str, Any]) -> float:
        """
        Compute probability of stress using exponential sensitivity.
        
        Primary contributors:
        - EDA-driven stress: stress_eda = 1 - exp(-k_eda * eda_mean), k_eda ~ 3-5
        - Environment modifiers: extreme temp/humidity/aqi
        - Task modifiers: high complexity/difficulty
        - System stress: CPU/memory/battery issues
        
        Returns:
            p_stress in [0, 1]
        """
        # Exponential sensitivity constants
        K_EDA = 3.0  # EDA stress coefficient
        K_BVP = 2.0  # BVP volatility coefficient
        
        stress_components = []
        weights = []
        
        # 1. EDA-driven stress (exponential)
        if 'eda_mean' in features:
            eda_raw = max(0.0, features['eda_mean'])
            stress_eda = 1.0 - math.exp(-K_EDA * eda_raw)
            stress_components.append(stress_eda)
            weights.append(0.5)  # Primary contributor
        
        # 2. BVP volatility (if available, adds to stress)
        if 'bvp_std' in features:
            bvp_std = max(0.0, features['bvp_std'])
            stress_bvp = 1.0 - math.exp(-K_BVP * bvp_std)
            stress_components.append(stress_bvp)
            weights.append(0.2)
        
        # 3. Environment stress (normalized, clipped)
        stress_env = 0.0
        if 'temp_c' in features:
            temp = features['temp_c']
            # Optimal: 20-25°C, stress increases outside this range
            if temp < 10 or temp > 35:
                stress_env = 0.4
            elif temp < 15 or temp > 30:
                stress_env = 0.2
            elif temp < 18 or temp > 28:
                stress_env = 0.1
        
        if 'humidity' in features:
            humidity = features['humidity']
            if humidity < 20 or humidity > 80:
                stress_env = max(stress_env, 0.2)
            elif humidity < 30 or humidity > 70:
                stress_env = max(stress_env, 0.1)
        
        if 'aqi' in features:
            aqi = features.get('aqi', 50)
            if aqi > 150:
                stress_env = max(stress_env, 0.3)
            elif aqi > 100:
                stress_env = max(stress_env, 0.15)
        
        if stress_env > 0:
            stress_components.append(stress_env)
            weights.append(0.15)
        
        # 4. Task stress
        stress_task = 0.0
        if 'task_complexity' in features:
            task_comp = features['task_complexity']
            stress_task = max(stress_task, task_comp * 0.3)
        if 'task_difficulty' in features:
            task_diff = features['task_difficulty']
            stress_task = max(stress_task, task_diff * 0.3)
        if 'task_stress' in features:
            task_stress = features['task_stress']
            stress_task = max(stress_task, task_stress * 0.4)
        
        if stress_task > 0:
            stress_components.append(stress_task)
            weights.append(0.1)
        
        # 5. System stress
        if 'system_stress' in features:
            system_stress = features['system_stress']
            stress_components.append(system_stress)
            weights.append(0.05)
        
        # Combine components with weights
        if stress_components and weights:
            total_weight = sum(weights)
            if total_weight > 0:
                p_stress = sum(s * w for s, w in zip(stress_components, weights)) / total_weight
            else:
                p_stress = 0.5
        else:
            # Fallback: use modality scores if no direct features
            weighted_stress = 0.0
            total_weight = 0.0
            for modality, score in scores.items():
                if modality in self.weights:
                    weight = self.weights[modality]
                    stress_contrib = (1.0 - score) * weight
                    weighted_stress += stress_contrib
                    total_weight += weight
            p_stress = weighted_stress / total_weight if total_weight > 0 else 0.5
        
        # Audio stress keywords boost
        if 'has_stress_keywords' in features and features['has_stress_keywords'] > 0.5:
            p_stress = min(1.0, p_stress + 0.15)
        
        return float(np.clip(p_stress, 0.0, 1.0))
    
    def _compute_p_focus(self, scores: Dict[str, float], features: Dict[str, Any]) -> float:
        """Compute probability of focus."""
        # Focus requires good environment, moderate stress, good system state
        env_score = scores.get('env', 0.5)
        system_score = scores.get('system', 0.5)
        physio_score = scores.get('physio', 0.5)
        
        # Focus = moderate stress (0.2-0.5) + good environment + good system
        p_stress = self._compute_p_stress(scores, features)
        
        if 0.2 <= p_stress <= 0.5 and env_score >= 0.8 and system_score >= 0.7:
            p_focus = (env_score * 0.4 + system_score * 0.3 + physio_score * 0.3)
        else:
            p_focus = 0.0
        
        return float(np.clip(p_focus, 0.0, 1.0))
    
    def _compute_p_chaos(self, scores: Dict[str, float], features: Dict[str, Any]) -> float:
        """
        Compute probability of chaos/overload using exponential sensitivity.
        
        Primary contributors:
        - Motion-driven chaos: chaos_motion = 1 - exp(-k_motion * acc_std), k_motion ~ 3-4
        - BVP volatility: erratic heart rate patterns
        - System issues: high error rates, network latency
        - Combined with high stress: chaos amplifies when stress is already high
        
        Returns:
            p_chaos in [0, 1]
        """
        # Exponential sensitivity constants
        K_MOTION = 4.0  # Motion chaos coefficient
        K_BVP_CHAOS = 3.0  # BVP chaos coefficient
        
        chaos_components = []
        weights = []
        
        # 1. Motion-driven chaos (exponential from acc_std)
        if 'acc_std' in features:
            acc_std = max(0.0, features['acc_std'])
            chaos_motion = 1.0 - math.exp(-K_MOTION * acc_std)
            chaos_components.append(chaos_motion)
            weights.append(0.6)  # Primary contributor
        
        # 2. BVP volatility (erratic heart rate = chaos)
        if 'bvp_std' in features:
            bvp_std = max(0.0, features['bvp_std'])
            chaos_bvp = 1.0 - math.exp(-K_BVP_CHAOS * bvp_std)
            chaos_components.append(chaos_bvp)
            weights.append(0.2)
        
        # 3. System chaos (error rates, network issues)
        system_chaos = 0.0
        if 'system_stress' in features:
            system_chaos = features['system_stress'] * 0.5
        if 'error_rate' in features:
            error_rate = features.get('error_rate', 0.0)
            system_chaos = max(system_chaos, error_rate * 0.4)
        
        if system_chaos > 0:
            chaos_components.append(system_chaos)
            weights.append(0.2)
        
        # Combine components
        if chaos_components and weights:
            total_weight = sum(weights)
            if total_weight > 0:
                p_chaos = sum(c * w for c, w in zip(chaos_components, weights)) / total_weight
            else:
                p_chaos = 0.0
        else:
            # Fallback: derive from stress and motion if no direct features
            p_stress = self._compute_p_stress(scores, features)
            system_stress = features.get('system_stress', 0.0)
            motion_score = scores.get('motion', 0.5)
            p_chaos = p_stress * 0.4 + system_stress * 0.3 + (1.0 - motion_score) * 0.3
        
        # Amplify chaos when stress is already high (stress + chaos = overload)
        p_stress = self._compute_p_stress(scores, features)
        if p_stress > 0.7:
            # High stress amplifies chaos
            p_chaos = min(1.0, p_chaos + (p_stress - 0.7) * 0.5)
        
        return float(np.clip(p_chaos, 0.0, 1.0))
    
    def _compute_env_score(self, features: Dict[str, Any]) -> float:
        """Compute environmental comfort score."""
        if 'temp_c' not in features or 'humidity' not in features or 'aqi' not in features:
            return 0.5
        
        temp_c = features['temp_c']
        humidity = features['humidity']
        aqi = features.get('aqi', 50)
        
        # Comfortable temperature range: 20-25°C
        temp_score = 1.0
        if temp_c < 18 or temp_c > 28:
            temp_score = 0.5
        elif temp_c < 20 or temp_c > 25:
            temp_score = 0.7
        
        # Comfortable humidity: 40-60%
        humidity_score = 1.0
        if humidity < 30 or humidity > 70:
            humidity_score = 0.5
        elif humidity < 40 or humidity > 60:
            humidity_score = 0.7
        
        # Good AQI: < 50
        aqi_score = 1.0
        if aqi > 100:
            aqi_score = 0.3
        elif aqi > 50:
            aqi_score = 0.7
        
        return float((temp_score + humidity_score + aqi_score) / 3.0)
    
    def _compute_circadian_score(self, features: Dict[str, Any]) -> float:
        """Compute circadian alignment score."""
        if 'local_hour' not in features:
            return 0.5
        
        hour = int(features['local_hour'])
        
        # Optimal hours: 8-12 (morning focus), 14-18 (afternoon focus)
        if 8 <= hour <= 12 or 14 <= hour <= 18:
            return 1.0
        elif 6 <= hour <= 22:
            return 0.8
        else:
            return 0.5
    
    def _check_emergency_indicators(
        self, 
        features: Dict[str, Any], 
        request: CAVRequestV2
    ) -> Dict[str, bool]:
        """Check for emergency conditions."""
        indicators = {
            'has_emergency': False,
            'system_critical': False
        }
        
        # Check system stress
        if features.get('system_stress', 0.0) > 0.9:
            indicators['system_critical'] = True
            indicators['has_emergency'] = True
        
        # Check for emergency keywords in audio
        if request.audio and request.audio.keywords:
            emergency_keywords = ['emergency', 'help', 'stop', 'danger', 'critical']
            if any(kw.lower() in emergency_keywords for kw in request.audio.keywords):
                indicators['has_emergency'] = True
        
        # Check for very high stress
        p_stress = self._compute_p_stress(
            self._compute_modality_scores(features, {}, []),
            features
        )
        if p_stress > 0.95:
            indicators['has_emergency'] = True
        
        return indicators
    
    def _compute_cav_vector(
        self, 
        scores: Dict[str, float], 
        p_stress: float, 
        p_focus: float, 
        p_chaos: float,
        features: Dict[str, Any] = None
    ) -> List[float]:
        """Compute multidimensional CAV vector."""
        if features is None:
            features = {}
        
        # Base CAV components
        bio_component = 1.0 - p_stress
        env_component = scores.get('env', 0.5)
        circadian_component = self._compute_circadian_score(features)
        
        # Multidimensional vector: [bio, env, circadian, focus, chaos, system]
        cav_vector = [
            float(bio_component),
            float(env_component),
            float(circadian_component),
            float(p_focus),
            float(p_chaos),
            float(scores.get('system', 0.5))
        ]
        
        # Normalize to [0-1] range
        cav_vector = [max(0.0, min(1.0, v)) for v in cav_vector]
        
        return cav_vector
    
    def _compute_confidence(
        self, 
        modalities_present: List[str], 
        features: Dict[str, Any]
    ) -> float:
        """Compute overall confidence based on available data."""
        # More modalities = higher confidence
        num_modalities = len(modalities_present)
        base_confidence = min(1.0, 0.5 + num_modalities * 0.1)
        
        # More features = higher confidence
        num_features = len(features)
        feature_boost = min(0.3, num_features / 100.0)
        
        confidence = base_confidence + feature_boost
        return float(np.clip(confidence, 0.0, 1.0))

