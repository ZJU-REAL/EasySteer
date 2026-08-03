"""Bilingual (zh/en) user-facing messages shared by the API blueprints."""

MESSAGES = {
    'zh': {
        'missing_field': '缺少必填字段: {field}',
        'file_not_found': '文件不存在: {path}',
        'server_error': '服务器错误: {error}',
        'not_found': 'Steer Vector ID {id} 不存在',
        'deleted': 'Steer Vector {name} 已删除',
        'created': 'Steer Vector配置创建成功',
        'generation_error': '生成失败: {error}',
        'model_loading_error': '模型加载失败: {error}',
    },
    'en': {
        'missing_field': 'Missing required field: {field}',
        'file_not_found': 'File not found: {path}',
        'server_error': 'Server error: {error}',
        'not_found': 'Steer Vector ID {id} does not exist',
        'deleted': 'Steer Vector {name} has been deleted',
        'created': 'Steer Vector configuration created successfully',
        'generation_error': 'Generation failed: {error}',
        'model_loading_error': 'Model loading failed: {error}',
    },
}


def get_message(key, lang='zh', **kwargs):
    """Get a message in the specified language, formatted with kwargs."""
    messages = MESSAGES.get(lang, MESSAGES['zh'])
    return messages.get(key, key).format(**kwargs)


def lang(request):
    """Language code ('zh'/'en') from a Flask request's Accept-Language header."""
    return request.headers.get('Accept-Language', 'zh').split(',')[0].split('-')[0]
