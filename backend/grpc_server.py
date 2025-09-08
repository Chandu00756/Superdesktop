"""Minimal gRPC server scaffold for Omega nodes service.
This is a starting point. QUIC transport for gRPC requires additional libraries or a QUIC-to-gRPC proxy (Envoy/Traefik).
Use this scaffold to implement gRPC service logic; later we can wire QUIC via a proxy or add aioquic-based transport.
"""
import asyncio
import logging
from concurrent import futures

import grpc

# Generated classes would normally be imported from generated _pb2/_pb2_grpc modules.
# To generate Python stubs from `protos/nodes.proto` use:
#   python -m grpc_tools.protoc -I=./protos --python_out=. --grpc_python_out=. ./protos/nodes.proto

# Placeholder imports (replace with actual generated modules after codegen)
# from protos import nodes_pb2, nodes_pb2_grpc

class NodeServiceServicer:  # nodes_pb2_grpc.NodeServiceServicer
    def Register(self, request, context):
        # Implement registration logic that calls into backend API or DB
        logging.info(f"gRPC Register called for node {request.node.node_id}")
        # Placeholder response
        # return nodes_pb2.RegisterResponse(success=True, node_id=request.node.node_id, message='Registered')
        return None

    async def Stream(self, request_iterator, context):
        # Bidirectional streaming handler
        async for msg in request_iterator:
            logging.debug(f"Received stream message: {msg}")
            # Echo back
            yield msg


def serve(host='0.0.0.0', port=50051):
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
    # nodes_pb2_grpc.add_NodeServiceServicer_to_server(NodeServiceServicer(), server)
    server.add_insecure_port(f'{host}:{port}')
    server.start()
    logging.info(f"gRPC server started on {host}:{port} (insecure)")
    try:
        server.wait_for_termination()
    except KeyboardInterrupt:
        server.stop(0)


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    serve()
