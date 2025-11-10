import importlib

try:
    m = importlib.import_module('algo3_ABC.continuous.main')
    print('module_loaded', bool(m))
    print('has_artificial_bee_colony', hasattr(m, 'artificial_bee_colony'))
    print('module_file', getattr(m, '__file__', 'N/A'))
    for name in ('func_to_optimize','LB','UB','N','D','MaxGen','limit'):
        print(name, 'in module:', hasattr(m, name))
except Exception as e:
    print('import_error:', e)
