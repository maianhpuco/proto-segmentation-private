"""
=============================================================================
QUICK START SCRIPT FOR PROTOTYPE SEGMENTATION
=============================================================================

This script helps you get started quickly with prototype segmentation,
even if you don't have the full datasets downloaded yet.

🎯 OPTIONS:
1. Test with dummy data (no download required)
2. Try to download PASCAL VOC (smaller than Cityscapes)
3. Set up for Cityscapes (manual download required)
4. Run example usage to verify everything works

🚀 USAGE:
    python quick_start.py

This will guide you through the setup process step by step.
=============================================================================
"""

import os
import sys
import subprocess
import yaml
from pathlib import Path


def test_with_dummy_data():
    """Test the system with dummy data"""
    print("🧪 Testing with dummy data...")
    print("="*50)
    
    try:
        # Run the example usage script
        result = subprocess.run([sys.executable, "example_usage.py"], 
                              capture_output=True, text=True, cwd=".")
        
        if result.returncode == 0:
            print("✅ Dummy data test passed!")
            print("🎉 Your prototype segmentation setup is working correctly!")
            return True
        else:
            print("❌ Dummy data test failed!")
            print("Error output:", result.stderr)
            return False
            
    except Exception as e:
        print(f"❌ Error running dummy data test: {e}")
        return False


def try_download_voc():
    """Try to download PASCAL VOC dataset"""
    print("📥 Attempting to download PASCAL VOC dataset...")
    print("="*50)
    
    try:
        # Try the simple download first
        result = subprocess.run([sys.executable, "download_datasets.py", "voc"], 
                              capture_output=True, text=True, cwd=".")
        
        if result.returncode == 0:
            print("✅ VOC download successful!")
            print("🔧 Updating config for VOC...")
            update_config_for_voc()
            return True
        else:
            print("❌ VOC download failed!")
            print("Error:", result.stderr)
            return False
            
    except Exception as e:
        print(f"❌ Error downloading VOC: {e}")
        return False


def update_config_for_voc():
    """Update config.yaml for PASCAL VOC dataset"""
    try:
        config_path = "config.yaml"
        
        # Load current config
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        # Update for VOC
        config['dataset']['name'] = 'pascal'
        config['model']['num_classes'] = 21  # VOC has 21 classes
        
        # Save updated config
        with open(config_path, 'w') as f:
            yaml.dump(config, f, default_flow_style=False)
        
        print("✅ Config updated for PASCAL VOC (21 classes)")
        
    except Exception as e:
        print(f"❌ Error updating config: {e}")


def setup_for_cityscapes():
    """Set up for Cityscapes dataset"""
    print("🌆 Setting up for Cityscapes dataset...")
    print("="*50)
    
    try:
        # Run the Cityscapes downloader
        result = subprocess.run([sys.executable, "download_cityscapes.py"], 
                              capture_output=True, text=True, cwd=".")
        
        print("Cityscapes setup output:")
        print(result.stdout)
        
        if result.stderr:
            print("Errors:")
            print(result.stderr)
            
        return True
        
    except Exception as e:
        print(f"❌ Error setting up Cityscapes: {e}")
        return False


def check_current_setup():
    """Check what's currently available"""
    print("🔍 Checking current setup...")
    print("="*50)
    
    # Check for data directories
    data_dir = Path("./data")
    if data_dir.exists():
        print("✅ Data directory exists")
        
        # Check for VOC
        voc_dir = data_dir / "VOCdevkit"
        if voc_dir.exists():
            print("✅ PASCAL VOC dataset found")
        else:
            print("❌ PASCAL VOC dataset not found")
        
        # Check for Cityscapes
        cityscapes_dir = data_dir / "cityscapes"
        if cityscapes_dir.exists():
            print("✅ Cityscapes directory found")
            # Check for actual files
            train_images = cityscapes_dir / "leftImg8bit" / "train"
            if train_images.exists() and any(train_images.iterdir()):
                print("✅ Cityscapes training images found")
            else:
                print("❌ Cityscapes training images not found")
        else:
            print("❌ Cityscapes directory not found")
    else:
        print("❌ Data directory not found")
    
    # Check config
    config_path = Path("config.yaml")
    if config_path.exists():
        print("✅ Config file found")
        try:
            with open(config_path, 'r') as f:
                config = yaml.safe_load(f)
            dataset_name = config.get('dataset', {}).get('name', 'unknown')
            num_classes = config.get('model', {}).get('num_classes', 'unknown')
            print(f"   Dataset: {dataset_name}")
            print(f"   Classes: {num_classes}")
        except Exception as e:
            print(f"❌ Error reading config: {e}")
    else:
        print("❌ Config file not found")


def main():
    """Main function"""
    print("🚀 Prototype Segmentation Quick Start")
    print("="*60)
    
    while True:
        print("\nWhat would you like to do?")
        print("1. Test with dummy data (recommended first step)")
        print("2. Try to download PASCAL VOC dataset")
        print("3. Set up for Cityscapes dataset")
        print("4. Check current setup")
        print("5. Exit")
        
        try:
            choice = input("\nEnter your choice (1-5): ").strip()
            
            if choice == "1":
                print("\n" + "="*60)
                success = test_with_dummy_data()
                if success:
                    print("\n🎉 Great! Your setup is working. You can now:")
                    print("   - Try downloading a real dataset (option 2 or 3)")
                    print("   - Modify config.yaml for your needs")
                    print("   - Run python train.py to start training")
                
            elif choice == "2":
                print("\n" + "="*60)
                success = try_download_voc()
                if success:
                    print("\n🎉 VOC dataset ready! You can now:")
                    print("   - Run python train.py to start training")
                    print("   - The config has been updated for VOC")
                
            elif choice == "3":
                print("\n" + "="*60)
                setup_for_cityscapes()
                
            elif choice == "4":
                print("\n" + "="*60)
                check_current_setup()
                
            elif choice == "5":
                print("👋 Goodbye!")
                break
                
            else:
                print("❌ Invalid choice. Please enter 1-5.")
                
        except KeyboardInterrupt:
            print("\n👋 Goodbye!")
            break
        except Exception as e:
            print(f"❌ Error: {e}")


if __name__ == "__main__":
    main()
