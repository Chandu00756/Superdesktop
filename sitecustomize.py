# sitecustomize.py - executed at interpreter startup when on sys.path
# Ensures protobuf uses the pure-Python implementation to avoid descriptor creation errors
import os
os.environ.setdefault('PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION', 'python')
