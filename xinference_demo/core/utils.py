import logging
import os
import orjson
from pydantic import BaseModel

logger = logging.getLogger(__name__)

def json_dumps(o):
    '''
        将 Python 结构体 dump 为 json 格式
    '''
    def _default(obj):
        if isinstance(obj, BaseModel):
            # return obj.dict()
            return obj.model_dump()
        raise TypeError

    return orjson.dumps(o, default=_default)

def log_async(logger):
    '''
    异步调用装饰器
    在 Debug 模式下,打印函数名称、参数、消耗时间等信息
    与同步调用装饰器的区别在于, 调用被装饰函数时添加 await 关键字
    '''
    import time
    from functools import wraps

    def decorator(func):
        @wraps(func)
        async def wrapped(*args, **kwargs):
            logger.debug(f"Enter {func.__name__}, args: {args}, kwargs: {kwargs}")
            start = time.time()
            ret = await func(*args, **kwargs)
            logger.debug(
                f"Leave {func.__name__}, elapsed time: {int(time.time() - start)} s"
            )
            return ret

        return wrapped

    return decorator

def log_sync(logger):
    '''
    同步调用装饰器
    在 Debug 模式下,打印函数名称、参数、消耗时间等信息
    '''
    import time
    from functools import wraps

    def decorator(func):
        @wraps(func)
        def wrapped(*args, **kwargs):
            logger.debug(f"Enter {func.__name__}, args: {args}, kwargs: {kwargs}")
            start = time.time()
            ret = func(*args, **kwargs)
            logger.debug(
                f"Leave {func.__name__}, elapsed time: {int(time.time() - start)} s"
            )
            return ret

        return wrapped

    return decorator

def purge_dir(d):
    if not os.path.exists(d) or not os.path.isdir(d):
        return
    for name in os.listdir(d):
        subdir = os.path.join(d, name)
        try:
            if (os.path.islink(subdir) and not os.path.exists(subdir)) or (
                len(os.listdir(subdir)) == 0
            ):
                logger.info("Remove empty directory: %s", subdir)
                os.rmdir(subdir)
        except Exception:
            pass