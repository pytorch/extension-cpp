import torch
inside_virtual_environment = sys.prefix != sys.base_prefix
if inside_virtual_environment and os.name == 'nt':
    dll_dir = os.path.join(sys.prefix, 'Lib\\site-packages\\torch\\lib')
    handle = os.add_dll_directory(dll_dir)
from . import _C, ops
