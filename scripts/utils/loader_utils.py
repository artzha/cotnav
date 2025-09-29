def construct_filters(cfg):
    """Loops throgh and constructs filters"""
    filters = {}
    for filter_dict in cfg['filters']:
        name = filter_dict['name']
        filters[name] = {
            'type': filter_dict['type'],
            'params': filter_dict['params']
        }
    return filters