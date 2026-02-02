#!/usr/bin/env python3
"""Unit tests for EDON v2 components (no server required)."""

import sys
import numpy as np
from app.v2.schemas_v2 import (
    CAVRequestV2, PhysioInput, MotionInput, EnvInput,
    VisionInput, AudioInput, TaskInput, SystemInput
)
from app.v2.device_profiles import (
    DeviceProfile, get_profile, get_available_modalities,
    validate_request_for_profile
)
from app.v2.pca_fusion import PCAFusion
from app.v2.neural_head import NeuralHeadMLP
from app.v2.multimodal_fusion import fuse_multimodal_features
from app.v2.engine_v2 import CAVEngineV2


def test_device_profiles():
    """Test device profiles."""
    print("[Test 1] Device Profiles")
    
    # Test profile retrieval
    profile = get_profile("humanoid_full")
    assert profile is not None, "humanoid_full profile not found"
    print(f"   [OK] humanoid_full profile loaded")
    
    profile = get_profile("wearable_limited")
    assert profile is not None, "wearable_limited profile not found"
    print(f"   [OK] wearable_limited profile loaded")
    
    profile = get_profile("drone_nav")
    assert profile is not None, "drone_nav profile not found"
    print(f"   [OK] drone_nav profile loaded")
    
    # Test available modalities
    humanoid = get_profile("humanoid_full")
    available = get_available_modalities(humanoid)
    assert 'physio' in available
    assert 'vision' in available
    print(f"   [OK] Available modalities: {available}")
    
    wearable = get_profile("wearable_limited")
    available = get_available_modalities(wearable)
    assert 'physio' in available
    assert 'vision' not in available
    print(f"   [OK] Wearable modalities: {available}")
    


def test_schemas():
    """Test v2 schemas."""
    print("\n[Test 2] Schemas")
    
    # Test minimal request (env only)
    request = CAVRequestV2(
        env=EnvInput(temp_c=22.0, humidity=50.0, aqi=35, local_hour=14)
    )
    assert request.env is not None
    print(f"   [OK] Minimal request (env only) created")
    
    # Test full request
    request = CAVRequestV2(
        physio=PhysioInput(
            EDA=[0.1] * 240,
            TEMP=[36.5] * 240,
            BVP=[0.5] * 240,
            ACC_x=[0.0] * 240,
            ACC_y=[0.0] * 240,
            ACC_z=[1.0] * 240
        ),
        motion=MotionInput(velocity_magnitude=0.5, torque_mean=25.0),
        env=EnvInput(temp_c=22.0, humidity=50.0, aqi=35, local_hour=14),
        vision=VisionInput(embedding=[0.1] * 128, objects=["person"]),
        audio=AudioInput(embedding=[0.05] * 64, keywords=["calm"]),
        task=TaskInput(goal="operate", confidence=0.8),
        system=SystemInput(cpu_usage=0.3, memory_usage=0.4)
    )
    assert request.physio is not None
    assert request.vision is not None
    print(f"   [OK] Full multimodal request created")
    
    # Test validation (empty request should fail)
    try:
        request = CAVRequestV2()
        assert False, "Empty request should fail validation"
    except ValueError:
        print(f"   [OK] Empty request correctly rejected")
    


def test_multimodal_fusion():
    """Test multimodal fusion."""
    print("\n[Test 3] Multimodal Fusion")
    
    request = CAVRequestV2(
        physio=PhysioInput(
            EDA=[0.1 + 0.01 * i for i in range(240)],
            TEMP=[36.5] * 240,
            BVP=[0.5] * 240,
            ACC_x=[0.0] * 240,
            ACC_y=[0.0] * 240,
            ACC_z=[1.0] * 240
        ),
        env=EnvInput(temp_c=22.0, humidity=50.0, aqi=35, local_hour=14),
        motion=MotionInput(velocity_magnitude=0.5, torque_mean=25.0)
    )
    
    fused = fuse_multimodal_features(request)
    assert 'features' in fused
    assert 'embeddings' in fused
    assert 'modalities_present' in fused
    
    features = fused['features']
    assert 'eda_mean' in features
    assert 'temp_c' in features
    assert 'velocity_mag' in features
    
    print(f"   [OK] Features extracted: {len(features)} features")
    print(f"   [OK] Modalities: {fused['modalities_present']}")
    


def test_pca_fusion():
    """Test PCA fusion."""
    print("\n[Test 4] PCA Fusion")
    
    # Create sample feature vectors
    feature_vectors = [
        {'eda_mean': 0.1, 'bvp_mean': 0.5, 'temp_c': 22.0, 'humidity': 50.0},
        {'eda_mean': 0.2, 'bvp_mean': 0.6, 'temp_c': 23.0, 'humidity': 55.0},
        {'eda_mean': 0.15, 'bvp_mean': 0.55, 'temp_c': 22.5, 'humidity': 52.0},
    ]
    
    pca = PCAFusion(n_components=128)
    pca.fit(feature_vectors)
    
    # Transform a single feature vector
    embedding = pca.transform(feature_vectors[0])
    assert len(embedding) == 128, f"Expected 128-dim, got {len(embedding)}"
    assert np.all(np.isfinite(embedding)), "Embedding contains NaN/Inf"
    
    # Check normalization
    norm = np.linalg.norm(embedding)
    assert abs(norm - 1.0) < 0.1, f"Embedding not normalized, norm={norm}"
    
    print(f"   [OK] PCA fitted and transformed")
    print(f"   [OK] Embedding shape: {embedding.shape}")
    print(f"   [OK] Embedding norm: {norm:.4f}")
    


def test_neural_head():
    """Test neural head."""
    print("\n[Test 5] Neural Head")
    
    # Create a 128-dim embedding
    embedding = np.random.randn(128).astype(np.float32)
    embedding = embedding / np.linalg.norm(embedding)  # Normalize
    
    neural_head = NeuralHeadMLP(input_dim=128, use_torch=False)  # Use numpy for testing
    
    prediction = neural_head.predict(embedding)
    
    assert 'state_class' in prediction
    assert 'state_probs' in prediction
    assert 'action_recommendations' in prediction
    assert 'confidence' in prediction
    
    assert prediction['state_class'] in ['restorative', 'focus', 'balanced', 'overload', 'alert', 'emergency']
    
    actions = prediction['action_recommendations']
    assert 'speed_scale' in actions
    assert 'torque_scale' in actions
    assert 'safety_scale' in actions
    assert 0.0 <= actions['speed_scale'] <= 2.0
    assert 0.0 <= actions['safety_scale'] <= 1.0
    
    print(f"   [OK] Neural head prediction successful")
    print(f"   [OK] State: {prediction['state_class']}")
    print(f"   [OK] Confidence: {prediction['confidence']:.3f}")
    print(f"   [OK] Speed scale: {actions['speed_scale']:.2f}")
    


def test_engine_v2():
    """Test v2 engine."""
    print("\n[Test 6] CAV Engine v2")
    
    engine = CAVEngineV2()
    
    # Create test request
    request = CAVRequestV2(
        physio=PhysioInput(
            EDA=[0.1] * 240,
            TEMP=[36.5] * 240,
            BVP=[0.5] * 240,
            ACC_x=[0.0] * 240,
            ACC_y=[0.0] * 240,
            ACC_z=[1.0] * 240
        ),
        env=EnvInput(temp_c=22.0, humidity=50.0, aqi=35, local_hour=14),
        motion=MotionInput(velocity_magnitude=0.5, torque_mean=25.0)
    )
    
    result = engine.compute_cav_v2(request)
    
    assert 'cav_vector' in result
    assert len(result['cav_vector']) == 128, f"Expected 128-dim, got {len(result['cav_vector'])}"
    assert 'state_class' in result
    assert 'p_stress' in result
    assert 'influences' in result
    assert 'confidence' in result
    
    print(f"   [OK] Engine computation successful")
    print(f"   [OK] CAV Vector: {len(result['cav_vector'])}-dim")
    print(f"   [OK] State: {result['state_class']}")
    print(f"   [OK] P-Stress: {result['p_stress']:.3f}")
    print(f"   [OK] Confidence: {result['confidence']:.3f}")
    
    # Test with device profile
    request.device_profile = "humanoid_full"
    result2 = engine.compute_cav_v2(request, device_profile="humanoid_full")
    assert result2['state_class'] in ['restorative', 'focus', 'balanced', 'overload', 'alert', 'emergency']
    print(f"   [OK] Engine with device profile successful")
    


def main():
    """Run all unit tests."""
    print("=" * 70)
    print("EDON v2 Unit Tests")
    print("=" * 70)
    
    tests = [
        test_device_profiles,
        test_schemas,
        test_multimodal_fusion,
        test_pca_fusion,
        test_neural_head,
        test_engine_v2
    ]
    
    passed = 0
    failed = 0
    
    for test in tests:
        try:
            if test():
                passed += 1
            else:
                failed += 1
                print(f"   [FAIL] {test.__name__}")
        except Exception as e:
            failed += 1
            print(f"   [FAIL] {test.__name__}: {e}")
            import traceback
            traceback.print_exc()
    
    print("\n" + "=" * 70)
    print(f"Results: {passed} passed, {failed} failed")
    print("=" * 70)
    
    return failed == 0


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)

