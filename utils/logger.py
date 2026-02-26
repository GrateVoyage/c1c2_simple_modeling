"""
日志配置
"""
import logging

def setup_logger(name: str = __name__, level: int = logging.INFO) -> logging.Logger:
    """设置日志记录器"""
    logging.basicConfig(level=level, format='%(message)s')
    return logging.getLogger(name)

logger = setup_logger()
