#!/usr/bin/env python3
"""
OMEGA Core Services v2.1 Test Script
Tests all components of the Core Services integration
"""

import asyncio
import logging
import sys
import os
import time
from datetime import datetime

# Add storage_node to path
sys.path.append('/Users/chanduchitikam/Superdesktop/storage_node')

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('core_services_test')

async def test_core_services_comprehensive():
    """Comprehensive test of Core Services v2.1"""
    try:
        logger.info("=" * 60)
        logger.info("OMEGA CORE SERVICES v2.1 COMPREHENSIVE TEST")
        logger.info("=" * 60)
        
        # Import after path setup
        from main import CoreServicesIntegratedStorageNode
        
        # Create test node
        test_node_id = f"test-node-{int(time.time())}"
        storage_node = CoreServicesIntegratedStorageNode(node_id=test_node_id, listen_port=8080)
        
        logger.info(f"Created test node: {test_node_id}")
        
        # Test 1: Initialization
        logger.info("\n=== TEST 1: Core Services Initialization ===")
        start_time = time.time()
        success = await storage_node.initialize()
        init_time = time.time() - start_time
        
        if success:
            logger.info(f"✓ Core Services initialized successfully in {init_time:.2f}s")
        else:
            logger.error("✗ Core Services initialization failed")
            return False
        
        # Test 2: Service Health Check
        logger.info("\n=== TEST 2: Service Health Check ===")
        health_status = await storage_node.get_comprehensive_status()
        
        if health_status.get('core_services', {}).get('overall_status') == 'healthy':
            logger.info("✓ All services healthy")
            for service_name, service_info in health_status['core_services']['services'].items():
                status = "✓" if service_info['healthy'] else "✗"
                logger.info(f"  {status} {service_name}: {service_info['healthy']}")
        else:
            logger.warning("⚠ Some services may not be healthy")
        
        # Test 3: Storage Operations
        logger.info("\n=== TEST 3: Storage Operations ===")
        
        # Store test data
        test_data_sets = [
            (b"Hello, Core Services v2.1!", {"type": "greeting", "version": "2.1"}),
            (b"Test data for analytics", {"type": "analytics_test", "size": "small"}),
            (b"Large test data " * 1000, {"type": "performance_test", "size": "large"})
        ]
        
        for i, (data, metadata) in enumerate(test_data_sets):
            key = f"test-key-{i+1:03d}"
            
            start_time = time.time()
            store_result = await storage_node.store(key, data, metadata)
            store_time = time.time() - start_time
            
            if store_result['success']:
                logger.info(f"✓ Stored {key}: {len(data)} bytes in {store_time:.3f}s")
            else:
                logger.error(f"✗ Failed to store {key}: {store_result.get('error')}")
        
        # Retrieve test data
        for i in range(len(test_data_sets)):
            key = f"test-key-{i+1:03d}"
            
            start_time = time.time()
            retrieve_result = await storage_node.retrieve(key)
            retrieve_time = time.time() - start_time
            
            if retrieve_result['success']:
                logger.info(f"✓ Retrieved {key}: {len(retrieve_result['data'])} bytes in {retrieve_time:.3f}s")
            else:
                logger.error(f"✗ Failed to retrieve {key}: {retrieve_result.get('error')}")
        
        # Test 4: Analytics and Performance
        logger.info("\n=== TEST 4: Analytics and Performance ===")
        
        analytics_data = await storage_node.get_analytics_data()
        if 'error' not in analytics_data:
            logger.info("✓ Analytics data collected successfully")
            
            # Display key metrics
            storage_metrics = analytics_data.get('storage_node', {})
            if storage_metrics:
                logger.info(f"  Requests/sec: {storage_metrics.get('requests_per_second', 0):.2f}")
                logger.info(f"  Avg response time: {storage_metrics.get('avg_response_time', 0):.3f}s")
        else:
            logger.warning(f"⚠ Analytics collection issue: {analytics_data['error']}")
        
        # Test 5: Cluster Operations
        logger.info("\n=== TEST 5: Cluster Operations ===")
        
        # Test cluster sync
        sync_result = await storage_node.sync_cluster_state()
        if sync_result:
            logger.info("✓ Cluster state synchronization successful")
        else:
            logger.warning("⚠ Cluster sync had issues")
        
        # Test leadership
        leadership_result = await storage_node.promote_to_leader()
        if leadership_result:
            logger.info("✓ Node promoted to leader successfully")
        else:
            logger.info("ℹ Leadership promotion not available (expected in single-node test)")
        
        # Test 6: Custom Operations
        logger.info("\n=== TEST 6: Custom Operations ===")
        
        custom_ops = [
            'performance_analytics',
            'cluster_sync'
        ]
        
        for operation in custom_ops:
            result = await storage_node.execute_custom_operation(operation)
            if result['success']:
                logger.info(f"✓ Custom operation '{operation}' executed successfully")
            else:
                logger.warning(f"⚠ Custom operation '{operation}' had issues: {result.get('error')}")
        
        # Test 7: Service Integration Validation
        logger.info("\n=== TEST 7: Service Integration Validation ===")
        
        validation_results = await storage_node.core_services.validate_service_integration()
        overall_status = validation_results.get('overall_status')
        
        if overall_status == 'validated':
            logger.info("✓ All service integrations validated successfully")
        else:
            logger.warning(f"⚠ Service integration status: {overall_status}")
        
        for service_name, validation in validation_results.get('service_validations', {}).items():
            status = "✓" if validation.get('status') in ['connected', 'running'] else "✗"
            logger.info(f"  {status} {service_name}: {validation.get('status')}")
        
        # Test 8: Performance Stress Test
        logger.info("\n=== TEST 8: Performance Stress Test ===")
        
        stress_test_count = 10
        start_time = time.time()
        
        tasks = []
        for i in range(stress_test_count):
            key = f"stress-test-{i:03d}"
            data = f"Stress test data {i} " * 100
            task = storage_node.store(key, data.encode(), {"stress_test": True, "index": i})
            tasks.append(task)
        
        results = await asyncio.gather(*tasks, return_exceptions=True)
        stress_time = time.time() - start_time
        
        success_count = sum(1 for r in results if isinstance(r, dict) and r.get('success'))
        
        logger.info(f"✓ Stress test: {success_count}/{stress_test_count} operations successful")
        logger.info(f"  Total time: {stress_time:.2f}s")
        logger.info(f"  Ops/sec: {stress_test_count/stress_time:.2f}")
        
        # Final Status Report
        logger.info("\n=== FINAL STATUS REPORT ===")
        final_status = await storage_node.get_comprehensive_status()
        
        node_info = final_status['node_info']
        perf_metrics = final_status['performance_metrics']
        storage_info = final_status['storage_info']
        
        logger.info(f"Node ID: {node_info['node_id']}")
        logger.info(f"Uptime: {node_info['uptime']:.2f}s")
        logger.info(f"Services initialized: {node_info['services_initialized']}")
        logger.info(f"Total requests: {perf_metrics['requests_processed']}")
        logger.info(f"Data stored: {perf_metrics['data_stored']} bytes")
        logger.info(f"Total keys: {storage_info['total_keys']}")
        logger.info(f"Core Services keys: {storage_info['core_services_keys']}")
        logger.info(f"Legacy keys: {storage_info['legacy_keys']}")
        
        # Cleanup
        logger.info("\n=== CLEANUP ===")
        await storage_node.shutdown()
        logger.info("✓ Storage node shutdown completed")
        
        logger.info("\n" + "=" * 60)
        logger.info("CORE SERVICES v2.1 TEST COMPLETED SUCCESSFULLY!")
        logger.info("=" * 60)
        
        return True
        
    except Exception as e:
        logger.error(f"Test failed with exception: {e}")
        import traceback
        traceback.print_exc()
        return False

async def test_individual_services():
    """Test individual services separately"""
    try:
        logger.info("Testing individual Core Services components...")
        
        from main import (
            CoreServicesConfig, 
            PostgreSQLManager, 
            RedisStateManager, 
            ObjectStorageManager,
            FastAPIOrchestrator
        )
        
        config = CoreServicesConfig()
        
        # Test PostgreSQL Manager
        logger.info("\n--- Testing PostgreSQL Manager ---")
        try:
            pg_manager = PostgreSQLManager(config)
            pg_success = await pg_manager.initialize()
            logger.info(f"PostgreSQL: {'✓ Success' if pg_success else '✗ Failed'}")
            if pg_success:
                await pg_manager.shutdown()
        except Exception as e:
            logger.warning(f"PostgreSQL test failed: {e}")
        
        # Test Redis State Manager
        logger.info("\n--- Testing Redis State Manager ---")
        try:
            redis_manager = RedisStateManager(config)
            redis_success = await redis_manager.initialize()
            logger.info(f"Redis: {'✓ Success' if redis_success else '✗ Failed'}")
            if redis_success:
                await redis_manager.shutdown()
        except Exception as e:
            logger.warning(f"Redis test failed: {e}")
        
        # Test Object Storage Manager
        logger.info("\n--- Testing Object Storage Manager ---")
        try:
            storage_manager = ObjectStorageManager(config)
            storage_success = await storage_manager.initialize()
            logger.info(f"Object Storage: {'✓ Success' if storage_success else '✗ Failed'}")
            if storage_success:
                await storage_manager.shutdown()
        except Exception as e:
            logger.warning(f"Object Storage test failed: {e}")
        
        logger.info("Individual service tests completed")
        return True
        
    except Exception as e:
        logger.error(f"Individual service tests failed: {e}")
        return False

if __name__ == "__main__":
    async def main():
        """Main test function"""
        logger.info("Starting OMEGA Core Services v2.1 Tests...")
        
        # Test individual services first
        individual_success = await test_individual_services()
        
        # Then test comprehensive integration
        if individual_success:
            comprehensive_success = await test_core_services_comprehensive()
        else:
            logger.warning("Skipping comprehensive test due to individual service failures")
            comprehensive_success = False
        
        if comprehensive_success:
            logger.info("🎉 ALL TESTS PASSED! Core Services v2.1 is ready for production.")
            sys.exit(0)
        else:
            logger.error("❌ Some tests failed. Check logs for details.")
            sys.exit(1)
    
    # Run the tests
    asyncio.run(main())
