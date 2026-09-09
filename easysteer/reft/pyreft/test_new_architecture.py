#!/usr/bin/env python3
"""
Import smoke checks for EasySteer's bundled ReFT package.
"""

def test_core_imports():
    """Test importing core framework components"""
    try:
        from easysteer.reft.pyreft.core import IntervenableModel, IntervenableConfig
        from easysteer.reft.pyreft.core import TrainableIntervention, ConstantSourceIntervention
        print("✓ Core framework imports successful")
        return True
    except ImportError as e:
        print(f"✗ Core framework import failed: {e}")
        return False

def test_reft_imports():
    """Test importing REFT-specific components"""
    try:
        from easysteer.reft.pyreft.reft import ReftModel, ReftConfig
        from easysteer.reft.pyreft.reft import LoreftIntervention, NoreftIntervention
        print("✓ REFT module imports successful")
        return True
    except ImportError as e:
        print(f"✗ REFT module import failed: {e}")
        return False

def test_data_imports():
    """Test importing data processing components"""
    try:
        from easysteer.reft.pyreft.data import ReftDataset, ReftDataCollator
        print("✓ Data module imports successful")
        return True
    except ImportError as e:
        print(f"✗ Data module import failed: {e}")
        return False

def test_unified_imports():
    """Test importing from the unified package root"""
    try:
        from easysteer.reft import pyreft
        assert hasattr(pyreft, 'ReftModel')
        assert hasattr(pyreft, 'IntervenableModel')
        assert hasattr(pyreft, 'LoreftIntervention')
        print("✓ Unified package imports successful")
        return True
    except Exception as e:
        print(f"✗ Unified package import failed: {e}")
        return False

if __name__ == "__main__":
    print("Checking ReFT package imports...")
    print("=" * 50)

    tests = [
        test_core_imports,
        test_reft_imports,
        test_data_imports,
        test_unified_imports
    ]

    passed = 0
    for test in tests:
        if test():
            passed += 1

    print("=" * 50)
    print(f"Tests passed: {passed}/{len(tests)}")

    if passed == len(tests):
        print("All ReFT package imports passed.")
    else:
        print("⚠️  Some tests failed. Please check the import paths.")