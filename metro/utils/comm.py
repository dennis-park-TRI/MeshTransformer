from functools import wraps

from mpi4py import MPI

from detectron2.utils import comm as d2_comm


def is_distributed():
    return d2_comm.get_world_size() > 1


def mpi_broadcast(fn):
    """If distributed, only the master executes the function and broadcast the results to other workers.

    NOTE: not tested.

    Usage:
    @mpi_broadcast
    def foo(a, b): ...
    """
    @wraps(fn)
    def wrapper(*args, **kwargs):  # pylint: disable=unused-argument
        if not is_distributed():
            return fn(*args, **kwargs)

        if d2_comm.is_main_process():
            ret = fn(*args, **kwargs)
        else:
            ret = None
        ret = MPI.COMM_WORLD.bcast(ret, root=0)
        assert ret is not None
        return ret

    return wrapper
