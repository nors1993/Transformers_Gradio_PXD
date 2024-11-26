# Copyright Lightning AI. Licensed under the Apache License 2.0, see LICENSE file.

import logging
import re  # 引入正则表达式模块，用于处理字符串模式匹配。
from litgpt.model import GPT  # needs to be imported before config
from litgpt.config import Config
from litgpt.tokenizer import Tokenizer

# Suppress excessive warnings, see https://github.com/pytorch/pytorch/issues/111632
"""
这行代码编译了一个正则表达式，用于匹配包含“Profiler function ... will be ignored”的日志消息。
.* 表示任意字符的任意数量，允许该模式在消息的任意位置匹配。
"""
pattern = re.compile(".*Profiler function .* will be ignored")
"""
获取名为 "torch._dynamo.variables.torch" 的日志记录器，并添加一个过滤器。
过滤器使用一个匿名函数（lambda）来检查每个日志记录。
record.getMessage() 获取日志消息文本。
pattern.search(...) 检查该消息是否与编译的正则表达式匹配。
not 运算符用于确保只有不匹配此模式的日志消息才会通过过滤器。只有不包含“Profiler function ... will be ignored”的日志消息会被打印出来
"""
logging.getLogger("torch._dynamo.variables.torch").addFilter(
    lambda record: not pattern.search(record.getMessage())
)

# Avoid printing state-dict profiling output at the WARNING level when saving a checkpoint
# 禁用后，该日志记录器将不会输出任何日志消息。
logging.getLogger("torch.distributed.fsdp._optim_utils").disabled = True
logging.getLogger("torch.distributed.fsdp._debug_utils").disabled = True

"""
定义公共接口：
__all__ 是一个特殊变量，用于定义当使用 from module import * 语句时，哪些名称应该被导出。
在此例中，只有 GPT、Config 和 Tokenizer 这三个名称将被导出，其他未列出的名称将不会被导入。
"""
__all__ = ["GPT", "Config", "Tokenizer"]