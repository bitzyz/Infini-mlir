import os

def depth(seq):
    from collections.abc import Sequence
    from itertools import chain, count

    seq = iter(seq)
    try:
        for level in count():
            seq = chain([next(seq)], seq)
            seq = chain.from_iterable(s for s in seq if isinstance(s, Sequence))
    except StopIteration:
        return level

def str2shape(v):
    _shape = eval(v)
    if not isinstance(_shape, list):
        raise KeyError("not shape list:{}".format(v))
    if len(_shape) == 0:
        return []
    dim = depth(_shape)
    if dim == 1:
        return [_shape]
    if dim != 2:
        raise KeyError("not shape list:{}".format(v))
    return _shape

def str2list(v):
    vars = v.split(',')
    vars = [s.strip() for s in vars]
    while vars.count('') > 0:
        vars.remove('')
    return vars

def infini_opt_options():
    options = ["--shape-infer"]
    return options

def _os_system(cmd: list):
    cmd_str = " ".join(cmd)
    
    print("[Running]: {}".format(cmd_str))
    ret = os.system(cmd_str)
    if ret == 0:
        print("[Success]: {}".format(cmd_str))
    else:
        raise RuntimeError("[!Error]: {}".format(cmd_str))

def mlir_opt(mlirfile: str,
            opt_mlirfile: str):
    cmd = ["infini-opt", mlirfile]
    options = infini_opt_options()
    cmd.extend(options)
    cmd.extend(["-o", opt_mlirfile])
    _os_system(cmd)

