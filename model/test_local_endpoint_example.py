"""
Example script to demonstrate how to test a local endpoint.
This script shows how to use test_local_endpoint.py to deploy and test a model locally.
"""

import subprocess
import os
import argparse

def main():
    parser = argparse.ArgumentParser(description="Example script to test a local endpoint")
    parser.add_argument("--subscription-id", required=True, help="Azure subscription ID")
    parser.add_argument("--resource-group", required=True, help="Azure resource group name")
    parser.add_argument("--workspace-name", required=True, help="Azure ML workspace name")
    args = parser.parse_args()
    
    # Step 1: Generate a test image
    print("Step 1: Generating a test image...")
    subprocess.run(["python", "generate_test_image.py", "--digit", "5"], check=True)
    
    # Find the generated image
    test_images_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "test_images")
    test_image = None
    for file in os.listdir(test_images_dir):
        if file.startswith("mnist_digit_5_"):
            test_image = os.path.join(test_images_dir, file)
            break
    
    if not test_image:
        raise Exception("Could not find generated test image")
    
    print(f"Generated test image: {test_image}")
    
    # Step 2: Deploy and test the model on a local endpoint
    print("\nStep 2: Deploying and testing the model on a local endpoint...")
    cmd = [
        "python", "test_local_endpoint.py",
        "--subscription-id", args.subscription_id,
        "--resource-group", args.resource_group,
        "--workspace-name", args.workspace_name,
        "--test-image", test_image,
        "--debug"
    ]
    
    subprocess.run(cmd, check=True)
    
    # Step 3: Clean up the local endpoint
    print("\nStep 3: Cleaning up the local endpoint...")
    cleanup_cmd = [
        "python", "test_local_endpoint.py",
        "--subscription-id", args.subscription_id,
        "--resource-group", args.resource_group,
        "--workspace-name", args.workspace_name,
        "--cleanup"
    ]
    
    subprocess.run(cleanup_cmd, check=True)
    
    print("\nTest completed successfully!")

if __name__ == "__main__":
    main()
