#!/usr/bin/env python3
"""
Pipeline Testing and Validation Script

This script provides comprehensive testing functionality for the multi-feature pipeline
including system validation, component testing, and integration tests.

Usage:
    python test_pipeline.py --test_all
    python test_pipeline.py --test_memory
    python test_pipeline.py --test_dependencies
"""

import os
import sys
import subprocess
import psutil
import json
import tempfile
import shutil
from pathlib import Path
import argparse

class PipelineTester:
    """Comprehensive testing suite for the multi-feature pipeline"""
    
    def __init__(self):
        self.test_results = {}
        self.temp_dir = None
    
    def setup_test_environment(self):
        """Setup temporary test environment"""
        self.temp_dir = tempfile.mkdtemp(prefix="pipeline_test_")
        print(f"Test environment created: {self.temp_dir}")
        return True
    
    def cleanup_test_environment(self):
        """Cleanup test environment"""
        if self.temp_dir and os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)
            print(f"Test environment cleaned up: {self.temp_dir}")
    
    def test_system_requirements(self):
        """Test system requirements and resources"""
        print("Testing system requirements...")
        
        results = {}
        
        # Test Python version
        python_version = sys.version_info
        results['python_version'] = f"{python_version.major}.{python_version.minor}.{python_version.micro}"
        results['python_version_ok'] = python_version >= (3, 8)
        
        # Test memory
        memory = psutil.virtual_memory()
        results['total_memory_gb'] = round(memory.total / (1024**3), 1)
        results['available_memory_gb'] = round(memory.available / (1024**3), 1)
        results['memory_sufficient'] = memory.total >= 8 * (1024**3)  # 8GB minimum
        
        # Test disk space
        disk = psutil.disk_usage('.')
        results['disk_free_gb'] = round(disk.free / (1024**3), 1)
        results['disk_sufficient'] = disk.free >= 50 * (1024**3)  # 50GB minimum
        
        # Test CUDA availability
        try:
            import torch
            results['cuda_available'] = torch.cuda.is_available()
            if torch.cuda.is_available():
                results['cuda_device_count'] = torch.cuda.device_count()
                results['cuda_device_name'] = torch.cuda.get_device_name(0)
        except ImportError:
            results['cuda_available'] = False
            results['torch_available'] = False
        
        self.test_results['system'] = results
        return all([
            results['python_version_ok'],
            results['memory_sufficient'],
            results['disk_sufficient']
        ])
    
    def test_dependencies(self):
        """Test required Python dependencies"""
        print("Testing Python dependencies...")
        
        required_packages = [
            'numpy', 'pandas', 'scikit-learn', 'torch', 'transformers',
            'peft', 'datasets', 'accelerate', 'bitsandbytes', 'psutil',
            'tqdm', 'matplotlib', 'seaborn', 'scipy', 'huggingface_hub'
        ]
        
        results = {}
        all_available = True
        
        for package in required_packages:
            try:
                # Handle special cases
                if package == 'huggingface_hub':
                    import huggingface_hub
                elif package == 'scikit-learn':
                    import sklearn
                else:
                    __import__(package)
                results[package] = {'available': True, 'version': 'unknown'}
            except ImportError as e:
                results[package] = {'available': False, 'error': str(e)}
                all_available = False
        
        self.test_results['dependencies'] = results
        return all_available
    
    def test_file_structure(self):
        """Test required file structure"""
        print("Testing file structure...")
        
        required_files = [
            'orchestrate_pipeline.py',
            'data_preprocessing.py',
            'feature_engineering.py',
            'llama_finetuning.py',
            'run_train.sh',
            'requirements.txt'
        ]
        
        results = {}
        all_present = True
        
        for file in required_files:
            exists = os.path.exists(file)
            results[file] = {'exists': exists}
            if exists:
                results[file]['size'] = os.path.getsize(file)
            else:
                all_present = False
        
        # Check for directories
        required_dirs = ['data', 'logs', 'outputs']
        for dir_name in required_dirs:
            os.makedirs(dir_name, exist_ok=True)
            results[f"{dir_name}/"] = {'exists': True, 'created': True}
        
        self.test_results['file_structure'] = results
        return all_present
    
    def test_data_preprocessing_component(self):
        """Test data preprocessing component"""
        print("Testing data preprocessing component...")
        
        results = {}
        
        try:
            # Test import
            from data_preprocessing import NetworkTrafficDataProcessor
            results['import_success'] = True
            
            # Test basic functionality with minimal data
            test_csv_path = os.path.join(self.temp_dir, 'test_data.csv')
            self.create_test_csv(test_csv_path)
            
            processor = NetworkTrafficDataProcessor(test_csv_path, n_features=5)
            results['initialization_success'] = True
            
        except Exception as e:
            results['import_success'] = False
            results['error'] = str(e)
            return False
        
        self.test_results['data_preprocessing'] = results
        return results.get('initialization_success', False)
    
    def test_feature_engineering_component(self):
        """Test feature engineering component"""
        print("Testing feature engineering component...")
        
        results = {}
        
        try:
            # Test import
            from feature_engineering import CICIoTFeatureEngineer
            results['import_success'] = True
            
            # Test basic configuration
            class TestConfig:
                def __init__(self):
                    self.n_features_final = 5
                    self.n_features_statistical = 10
                    self.variance_threshold = 0.01
                    self.seed = 42
            
            config = TestConfig()
            engineer = CICIoTFeatureEngineer(config)
            results['initialization_success'] = True
            
        except Exception as e:
            results['import_success'] = False
            results['error'] = str(e)
            return False
        
        self.test_results['feature_engineering'] = results
        return results.get('initialization_success', False)
    
    def test_orchestration_component(self):
        """Test orchestration component"""
        print("Testing orchestration component...")
        
        results = {}
        
        try:
            # Test import and basic functionality
            from orchestrate_pipeline import MemoryManager, TaskExecutor, PipelineOrchestrator
            results['import_success'] = True
            
            # Test memory manager
            memory_manager = MemoryManager()
            resource_config = memory_manager.get_resource_config(1000)
            results['memory_manager_success'] = True
            results['max_concurrent_tasks'] = resource_config.max_concurrent_tasks
            
            # Test task executor
            task_executor = TaskExecutor()
            results['task_executor_success'] = True
            
        except Exception as e:
            results['import_success'] = False
            results['error'] = str(e)
            return False
        
        self.test_results['orchestration'] = results
        return results.get('task_executor_success', False)
    
    def test_dry_run_execution(self):
        """Test dry run execution"""
        print("Testing dry run execution...")
        
        results = {}
        
        try:
            # Create minimal test CSV
            test_csv_path = os.path.join(self.temp_dir, 'test_data.csv')
            self.create_test_csv(test_csv_path)
            
            # Test dry run
            cmd = [
                sys.executable, 'orchestrate_pipeline.py',
                '--features_list', '5',
                '--dry_run',
                '--per_feature_mem_mb', '500',
                '--per_train_mem_mb', '1000'
            ]
            
            process = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=30
            )
            
            results['dry_run_exit_code'] = process.returncode
            results['dry_run_success'] = process.returncode == 0
            results['stdout'] = process.stdout
            results['stderr'] = process.stderr
            
        except subprocess.TimeoutExpired:
            results['dry_run_success'] = False
            results['error'] = 'Dry run timed out'
        except Exception as e:
            results['dry_run_success'] = False
            results['error'] = str(e)
        
        self.test_results['dry_run'] = results
        return results.get('dry_run_success', False)
    
    def create_test_csv(self, csv_path):
        """Create a minimal test CSV for testing"""
        import pandas as pd
        import numpy as np
        
        # Create synthetic test data
        n_samples = 100
        data = {
            'flow_duration': np.random.randn(n_samples),
            'Header_Length': np.random.randn(n_samples),
            'Protocol Type': np.random.randint(1, 10, n_samples),
            'Duration': np.random.randn(n_samples),
            'Rate': np.random.randn(n_samples),
            'TCP': np.random.randint(0, 2, n_samples),
            'UDP': np.random.randint(0, 2, n_samples),
            'HTTP': np.random.randint(0, 2, n_samples),
            'label': np.random.choice(['Benign', 'Malicious'], n_samples)
        }
        
        df = pd.DataFrame(data)
        df.to_csv(csv_path, index=False)
        print(f"Test CSV created: {csv_path}")
    
    def generate_test_report(self):
        """Generate comprehensive test report"""
        print("\n" + "="*60)
        print("PIPELINE TEST REPORT")
        print("="*60)
        
        overall_success = True
        
        for test_category, results in self.test_results.items():
            print(f"\n{test_category.upper()} TEST RESULTS:")
            print("-" * 40)
            
            if test_category == 'system':
                print(f"Python Version: {results['python_version']} ({'✓' if results['python_version_ok'] else '✗'})")
                print(f"Total Memory: {results['total_memory_gb']} GB ({'✓' if results['memory_sufficient'] else '✗'})")
                print(f"Available Memory: {results['available_memory_gb']} GB")
                print(f"Disk Free: {results['disk_free_gb']} GB ({'✓' if results['disk_sufficient'] else '✗'})")
                print(f"CUDA Available: {'✓' if results.get('cuda_available', False) else '✗'}")
                
                if not (results['python_version_ok'] and results['memory_sufficient'] and results['disk_sufficient']):
                    overall_success = False
            
            elif test_category == 'dependencies':
                failed_deps = [pkg for pkg, info in results.items() if not info['available']]
                if failed_deps:
                    print(f"Missing dependencies: {', '.join(failed_deps)}")
                    overall_success = False
                else:
                    print("All dependencies available ✓")
            
            elif test_category == 'file_structure':
                missing_files = [file for file, info in results.items() if not info['exists']]
                if missing_files:
                    print(f"Missing files: {', '.join(missing_files)}")
                    overall_success = False
                else:
                    print("All required files present ✓")
            
            else:
                # Component tests
                success = results.get('import_success', False) and not 'error' in results
                status = '✓' if success else '✗'
                print(f"Component test: {status}")
                
                if 'error' in results:
                    print(f"Error: {results['error']}")
                    overall_success = False
        
        print("\n" + "="*60)
        print(f"OVERALL TEST STATUS: {'✓ PASSED' if overall_success else '✗ FAILED'}")
        print("="*60)
        
        if not overall_success:
            print("\nRecommendations:")
            if not self.test_results.get('system', {}).get('memory_sufficient', True):
                print("- Increase system memory or reduce per-task memory allocation")
            if not self.test_results.get('system', {}).get('disk_sufficient', True):
                print("- Free up disk space or use external storage")
            print("- Check missing dependencies and install with: pip install -r requirements.txt")
            print("- Ensure all required files are present in the working directory")
        
        return overall_success
    
    def run_all_tests(self):
        """Run all pipeline tests"""
        print("Running comprehensive pipeline tests...")
        
        self.setup_test_environment()
        
        try:
            # Run all test components
            self.test_system_requirements()
            self.test_dependencies()
            self.test_file_structure()
            self.test_data_preprocessing_component()
            self.test_feature_engineering_component()
            self.test_orchestration_component()
            self.test_dry_run_execution()
            
            # Generate report
            success = self.generate_test_report()
            return success
            
        finally:
            self.cleanup_test_environment()

def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(
        description="Test and validate the multi-feature pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument(
        '--test_all',
        action='store_true',
        help='Run all pipeline tests'
    )
    
    parser.add_argument(
        '--test_system',
        action='store_true',
        help='Test system requirements only'
    )
    
    parser.add_argument(
        '--test_dependencies',
        action='store_true',
        help='Test Python dependencies only'
    )
    
    parser.add_argument(
        '--test_components',
        action='store_true',
        help='Test pipeline components only'
    )
    
    return parser.parse_args()

def main():
    """Main entry point for testing"""
    try:
        args = parse_arguments()
        tester = PipelineTester()
        
        if args.test_all or (not args.test_system and not args.test_dependencies and not args.test_components):
            success = tester.run_all_tests()
        elif args.test_system:
            tester.test_system_requirements()
            success = tester.generate_test_report()
        elif args.test_dependencies:
            tester.test_dependencies()
            success = tester.generate_test_report()
        elif args.test_components:
            tester.setup_test_environment()
            try:
                tester.test_data_preprocessing_component()
                tester.test_feature_engineering_component()
                tester.test_orchestration_component()
                success = tester.generate_test_report()
            finally:
                tester.cleanup_test_environment()
        
        sys.exit(0 if success else 1)
        
    except KeyboardInterrupt:
        print("\nTesting interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"Testing failed with error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()