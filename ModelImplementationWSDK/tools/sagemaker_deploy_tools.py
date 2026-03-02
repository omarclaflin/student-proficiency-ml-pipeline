#Boto3/Sagemaker update interaction functions
#Updating, scaling, deregister scaling, health checks, rollbacks

import boto3
import time
import sagemaker
from sagemaker import get_execution_role
import json
from datetime import datetime
import re
from botocore.exceptions import ClientError

# Initialize the SageMaker client
sagemaker_client = boto3.client('sagemaker')



def deregister_scalable_target(endpoint_name):
    client = boto3.client('application-autoscaling')
    resource_id = f'endpoint/{endpoint_name}/variant/AllTraffic'
    
    try:
        client.deregister_scalable_target(
            ServiceNamespace='sagemaker',
            ResourceId=resource_id,
            ScalableDimension='sagemaker:variant:DesiredInstanceCount'
        )
        print(f"Deregistered scalable target for endpoint: {endpoint_name}")
    except client.exceptions.ObjectNotFoundException:
        print(f"Scalable target for endpoint {endpoint_name} not found. Proceeding with update.")


def register_scalable_target(endpoint_name, min_capacity, max_capacity):
    client = boto3.client('application-autoscaling')
    resource_id = f'endpoint/{endpoint_name}/variant/AllTraffic'
    
    client.register_scalable_target(
        ServiceNamespace='sagemaker',
        ResourceId=resource_id,
        ScalableDimension='sagemaker:variant:DesiredInstanceCount',
        MinCapacity=min_capacity,
        MaxCapacity=max_capacity
    )
    print(f"Registered scalable target for endpoint: {endpoint_name}")

#
#this could be redundant to the register_scalable_target() but sets more params for scaling    
def setup_auto_scaling(endpoint_name, min_capacity, max_capacity, target_value, 
                      scale_in_cooldown=300, scale_out_cooldown=300):
    
    # Check status first
    status = sagemaker_client.describe_endpoint(EndpointName=endpoint_name)['EndpointStatus']
    if status != 'InService':
        print(f"Cannot modify scaling policy - endpoint status is {status}. Please wait for endpoint to be InService.")
        return
    
    client = boto3.client('application-autoscaling')
    resource_id = f'endpoint/{endpoint_name}/variant/AllTraffic'
    
    # First, try to delete existing scaling policy
    try:
        client.delete_scaling_policy(
            PolicyName=f'ScalingPolicy-{endpoint_name}',
            ServiceNamespace='sagemaker',
            ResourceId=resource_id,
            ScalableDimension='sagemaker:variant:DesiredInstanceCount'
        )
        print(f"Deleted existing scaling policy for endpoint: {endpoint_name}")
    except client.exceptions.ObjectNotFoundException:
        print("No existing scaling policy found")
    
    # Deregister existing target
    try:
        client.deregister_scalable_target(
            ServiceNamespace='sagemaker',
            ResourceId=resource_id,
            ScalableDimension='sagemaker:variant:DesiredInstanceCount'
        )
        print("Deregistered existing scalable target")
    except client.exceptions.ObjectNotFoundException:
        print("No existing scalable target found")
        
    # Register new scalable target
    client.register_scalable_target(
        ServiceNamespace='sagemaker',
        ResourceId=resource_id,
        ScalableDimension='sagemaker:variant:DesiredInstanceCount',
        MinCapacity=min_capacity,
        MaxCapacity=max_capacity
    )
    
    # Apply new scaling policy
    client.put_scaling_policy(
        PolicyName=f'ScalingPolicy-{endpoint_name}',
        ServiceNamespace='sagemaker',
        ResourceId=resource_id,
        ScalableDimension='sagemaker:variant:DesiredInstanceCount',
        PolicyType='TargetTrackingScaling',
        TargetTrackingScalingPolicyConfiguration={
            'TargetValue': target_value,
            'PredefinedMetricSpecification': {
                'PredefinedMetricType': 'SageMakerVariantInvocationsPerInstance'
            },
            'ScaleInCooldown': scale_in_cooldown,
            'ScaleOutCooldown': scale_out_cooldown
        }
    )

    print(f"Auto-scaling has been set up for endpoint: {endpoint_name}")
    print(f"Min capacity: {min_capacity}")
    print(f"Max capacity: {max_capacity}")
    print(f"Target value: {target_value} invocations per instance per minute")
    print(f"Scale-in cooldown: {scale_in_cooldown} seconds")
    print(f"Scale-out cooldown: {scale_out_cooldown} seconds")
    
def update_endpoint(endpoint_name, new_image_uri, test_message, new_instance_type=None):
    sagemaker_client = boto3.client('sagemaker')

    # Get the current endpoint configuration
    current_endpoint_config = sagemaker_client.describe_endpoint(EndpointName=endpoint_name)['EndpointConfigName']

    # Get the current model name and image
    current_config = sagemaker_client.describe_endpoint_config(EndpointConfigName=current_endpoint_config)
    current_model_name = current_config['ProductionVariants'][0]['ModelName']
    current_model = sagemaker_client.describe_model(ModelName=current_model_name)
    current_image_uri = current_model['PrimaryContainer']['Image']

    print(f"Current Endpoint Config: {current_endpoint_config}")
    print(f"Current Model Name: {current_model_name}")
    print(f"Current Image URI: {current_image_uri}")

    try:
        # Create a new model with the new image
        
        #update model name w current datetime, eg 'math-proficiency-model-2024-08-01-00-54-25-247'
        import re
        from datetime import datetime
        # Extract the base name (everything before the datetime)
        base_model_name = re.split(r'-\d{4}-', current_model_name)[0]
        new_model_name = '-'.join([base_model_name, datetime.now().strftime('%Y-%m-%d-%H-%M-%S-%f')[:-3]])

        sagemaker_client.create_model(
            ModelName=new_model_name,
            PrimaryContainer={
                'Image': new_image_uri,
            },
            ExecutionRoleArn=get_execution_role()
        )

        # Create a new endpoint configuration        
        import re
        from datetime import datetime
        base_config_name = current_endpoint_config.split('-')[0]  # Get just first part
        timestamp = datetime.now().strftime('%Y%m%d%H%M%S')  # Shorter timestamp format
        new_config_name = f"{base_config_name}-{timestamp}"
        
        #new_config_name = f"{current_endpoint_config}-{new_image_uri.split(':', 1)[-1]}"
        #determine instance type
        instance_type = new_instance_type if new_instance_type else current_config['ProductionVariants'][0]['InstanceType']


        sagemaker_client.create_endpoint_config(
            EndpointConfigName=new_config_name,
            ProductionVariants=[{
                'InstanceType': instance_type,
                'InitialInstanceCount': current_config['ProductionVariants'][0]['InitialInstanceCount'],
                'ModelName': new_model_name,
                'VariantName': 'AllTraffic'
            }]
        )
        print("New config name: ", new_config_name, " Old config name: ", current_endpoint_config)

        # Update the endpoint
        sagemaker_client.update_endpoint(
            EndpointName=endpoint_name,
            EndpointConfigName=new_config_name
        )

        print(f"Endpoint '{endpoint_name}' is being updated. This may take a few minutes...")
        waiter = sagemaker_client.get_waiter('endpoint_in_service')
        waiter.wait(EndpointName=endpoint_name)

        # Check if the new endpoint is healthy
        if check_endpoint_health(endpoint_name, test_message):
            print(f"Endpoint '{endpoint_name}' has been successfully updated with the new model.")
        else:
            raise Exception("New endpoint failed health check")
        
    #if fails, do a rollback
    except Exception as e:
        print(f"Update failed: {str(e)}. Rolling back to previous version...")

        # Rollback to the previous configuration
        sagemaker_client.update_endpoint(
            EndpointName=endpoint_name,
            EndpointConfigName=current_endpoint_config
        )

        print(f"Rolling back endpoint '{endpoint_name}'. This may take a few minutes...")
        waiter = sagemaker_client.get_waiter('endpoint_in_service')
        waiter.wait(EndpointName=endpoint_name)

        if check_endpoint_health(endpoint_name, test_message):
            print(f"Endpoint '{endpoint_name}' has been successfully rolled back to the previous version.")
        else:
            print(f"WARNING: Endpoint '{endpoint_name}' is not healthy after rollback. Manual intervention may be required.")
    
def check_endpoint_status(endpoint_name):
    """Check if endpoint is InService"""
    try:
        response = sagemaker_client.describe_endpoint(EndpointName=endpoint_name)
        status = response['EndpointStatus']
        print(f"Current endpoint status: {status}")
        return status == 'InService'
    except Exception as e:
        print(f"Error checking endpoint status: {str(e)}")
        return False

def update_endpoint_with_scaling(endpoint_name, new_image_uri, test_message, min_capacity, max_capacity, 
                                target_value=50, scale_in_cooldown=600, scale_out_cooldown=180, new_instance_type=None):
    try:
        # Check endpoint status first
        if not check_endpoint_status(endpoint_name):
            raise Exception("Endpoint is not InService. Please wait for current operations to complete.")
            
        print('Attempting deregistration of scalable target...')
        deregister_scalable_target(endpoint_name)
        print('Attempting updating of endpoint...')
        update_endpoint(endpoint_name, new_image_uri, test_message, new_instance_type)
    finally:
        print('Attempting registration of scaling of target...')
        setup_auto_scaling(
            endpoint_name=endpoint_name,
            min_capacity=min_capacity,
            max_capacity=max_capacity,
            target_value=target_value,
            scale_in_cooldown=scale_in_cooldown,
            scale_out_cooldown=scale_out_cooldown
        )

# Function to check if the endpoint exists
def endpoint_exists(endpoint_name):
    try:
        sagemaker_client.describe_endpoint(EndpointName=endpoint_name)
        return True
    except ClientError as e:
        if e.response['Error']['Code'] == 'ValidationException':
            return False
        else:
            raise e

def check_endpoint_health(endpoint_name, test_message):
    runtime = boto3.client('sagemaker-runtime')
    try:
        
        # Replace this with an actual inference request that suits your model
        response = runtime.invoke_endpoint(
            EndpointName=endpoint_name, 
            ContentType='application/json', 
            Body=json.dumps(test_message))
        # Replace this with an actual inference request that suits your model
        #response = runtime.invoke_endpoint(
        #    EndpointName=endpoint_name,
        #    ContentType='application/json',
        #    Body=json.dumps(test_message)
        #)
        return response['ResponseMetadata']['HTTPStatusCode'] == 200
    except Exception as e:
        print(f"Health check failed: {str(e)}")
        return False            
        
def get_previous_config(endpoint_name):
    sagemaker_client = boto3.client('sagemaker')
    
    # Get current endpoint configuration
    current_config = sagemaker_client.describe_endpoint(EndpointName=endpoint_name)['EndpointConfigName']
    base_config_name = current_config.split('-')[0]  # Get base name before timestamp
    
    # List all configs with this base name
    configs = sagemaker_client.list_endpoint_configs(NameContains=base_config_name)['EndpointConfigs']
    
    # Sort by name (which includes timestamp) in descending order
    sorted_configs = sorted([c['EndpointConfigName'] for c in configs], reverse=True)
    
    # Find current config and get the next one (previous version)
    try:
        current_index = sorted_configs.index(current_config)
        if current_index + 1 < len(sorted_configs):
            return sorted_configs[current_index + 1]  # Previous config
    except ValueError:
        pass
        
    return None

def rollback_to_previous(endpoint_name, test_message):
    sagemaker_client = boto3.client('sagemaker')
    
    # Get previous config
    previous_config = get_previous_config(endpoint_name)
    if not previous_config:
        print("No previous configuration found. Cannot rollback.")
        return False
        
    print(f"Found previous config: {previous_config}")
    
    try:
        # Deregister scaling if it exists
        print('Deregistering scalable target...')
        deregister_scalable_target(endpoint_name)
        
        # Update to previous config
        print(f"Rolling back to previous configuration...")
        sagemaker_client.update_endpoint(
            EndpointName=endpoint_name,
            EndpointConfigName=previous_config
        )
        
        print(f"Waiting for rollback to complete...")
        waiter = sagemaker_client.get_waiter('endpoint_in_service')
        waiter.wait(EndpointName=endpoint_name)
        
        # Check health
        if check_endpoint_health(endpoint_name, test_message):
            print(f"Endpoint successfully rolled back to previous version")
            return True
        else:
            print(f"Warning: Endpoint not healthy after rollback")
            return False
            
    except Exception as e:
        print(f"Rollback failed: {str(e)}")
        return False

    
    

def monitor_scaling():
    sagemaker = boto3.client('sagemaker')
    
    while True:
        response = sagemaker.describe_endpoint(
            EndpointName='[REDACTED]'
        )
        
        # Get endpoint config details to see instance type
        config_name = response['EndpointConfigName']
        config = sagemaker.describe_endpoint_config(EndpointConfigName=config_name)
        
        variant = response['ProductionVariants'][0]
        status = response['EndpointStatus']
        current = variant['CurrentInstanceCount']
        desired = variant['DesiredInstanceCount']
        instance_type = config['ProductionVariants'][0]['InstanceType']
        
        print(f"[{datetime.now().strftime('%H:%M:%S')}] Status: {status} | Instance Type: {instance_type} | Current Instances: {current} | Desired: {desired}")
        
        if status == 'InService' and current == desired:
            print("Scaling completed!")
            break
            
        time.sleep(30)  # Check every 30 seconds
        
        
def verify_scaling_policy(endpoint_name):
    client = boto3.client('application-autoscaling')
    sagemaker_client = boto3.client('sagemaker')
    resource_id = f'endpoint/{endpoint_name}/variant/AllTraffic'
    
    try:
        # Get endpoint configuration info
        endpoint_config_name = sagemaker_client.describe_endpoint(EndpointName=endpoint_name)['EndpointConfigName']
        endpoint_config = sagemaker_client.describe_endpoint_config(EndpointConfigName=endpoint_config_name)
        instance_type = endpoint_config['ProductionVariants'][0]['InstanceType']
        
        # Get scaling target info (min/max capacity)
        target_response = client.describe_scalable_targets(
            ServiceNamespace='sagemaker',
            ResourceIds=[resource_id],
            ScalableDimension='sagemaker:variant:DesiredInstanceCount'
        )
        
        # Get scaling policy info
        policy_response = client.describe_scaling_policies(
            ServiceNamespace='sagemaker',
            ResourceId=resource_id,
            ScalableDimension='sagemaker:variant:DesiredInstanceCount'
        )
        
        print(f"Current endpoint configuration:")
        print(f"Instance Type: {instance_type}")
        
        if target_response['ScalableTargets']:
            target = target_response['ScalableTargets'][0]
            print(f"Min Capacity: {target['MinCapacity']}")
            print(f"Max Capacity: {target['MaxCapacity']}")
        
        if policy_response['ScalingPolicies']:
            policy = policy_response['ScalingPolicies'][0]
            print(f"\nScaling policy configuration:")
            print(f"Policy Name: {policy['PolicyName']}")
            print(f"Target Value: {policy['TargetTrackingScalingPolicyConfiguration']['TargetValue']}")
            print(f"Scale-in Cooldown: {policy['TargetTrackingScalingPolicyConfiguration'].get('ScaleInCooldown')} seconds")
            print(f"Scale-out Cooldown: {policy['TargetTrackingScalingPolicyConfiguration'].get('ScaleOutCooldown')} seconds")
        else:
            print("\nNo scaling policies found")
            
    except Exception as e:
        print(f"Error checking scaling policy: {str(e)}")        
        
        
def check_scaling_activities(endpoint_name, MaxResults=50):
    client = boto3.client('application-autoscaling')
    resource_id = f'endpoint/{endpoint_name}/variant/AllTraffic'
    
    response = client.describe_scaling_activities(
        ServiceNamespace='sagemaker',
        ResourceId=resource_id,
        ScalableDimension='sagemaker:variant:DesiredInstanceCount',
        MaxResults=MaxResults
    )
    
    print("Recent scaling activities:")
    for activity in response['ScalingActivities']:
        print(f"\nActivity ID: {activity['ActivityId']}")
        print(f"Status: {activity['StatusCode']}")
        print(f"Description: {activity['Description']}")
        print(f"Cause: {activity['Cause']}")
        print(f"Start Time: {activity['StartTime']}")
        if activity.get('StatusMessage'):
            print(f"Status Message: {activity['StatusMessage']}")
            
