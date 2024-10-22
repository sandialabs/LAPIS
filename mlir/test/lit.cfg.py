import lit.formats

config.name = "LAPIS Dialect Tests"
config.test_format = lit.formats.ShTest(True)

config.suffixes = ['.mlir']

config.test_source_root = os.path.dirname(__file__)
config.test_exec_root = os.path.join(config.lapis_obj_root, 'mlir/test')
config.substitutions.append(('%lapis-opt', os.path.join(config.lapis_tool_bin, 'lapis-opt')))
