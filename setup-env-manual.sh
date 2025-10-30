#!/bin/bash

# FuseX 手动环境设置脚本
# 使用方法: source setup-env-manual.sh

echo "设置 FuseX 环境..."

# 设置 Python 路径
export PYTHONPATH="/home/zhihenc5/cur_research/FuseX/python:/home/zhihenc5/cur_research/FuseX/build:/home/zhihenc5/cur_research/FuseX/testing/mculib/python:/home/zhihenc5/cur_research/FuseX/testing/tileflow/python:$PYTHONPATH"

# 设置 C++ 包含路径
export CPLUS_INCLUDE_PATH="/home/zhihenc5/cur_research/FuseX/testing/mculib/include:$CPLUS_INCLUDE_PATH"

echo "✅ FuseX 环境设置完成"
echo "Python 路径: $PYTHONPATH"

# 验证模块是否可以导入
echo "验证模块导入..."
python3 -c "
try:
    import domino
    import dominoc
    import mculib
    import tileflow
    print('✅ 所有模块导入成功')
except ImportError as e:
    print(f'❌ 模块导入失败: {e}')
"