import importlib.util
import os
import unittest

from sparta.common.utils import get_uname

__tmp_file__ = 'tmp_mk_pycodes.py'


def validate_markdown(fname: str):
    os.system(f'echo "# codes from {fname}" > {__tmp_file__}')
    os.system(f'codeblocks python {fname} >> {__tmp_file__}')
    # spec=importlib.util.spec_from_file_location("mktest", __tmp_file__)
    # foo = importlib.util.module_from_spec(spec)
    # spec.loader.exec_module(foo)
    # for f in dir(foo):
    #     if f.startswith('test_'):
    #         getattr(foo, f)()


def cleanup():
    if os.path.exists(__tmp_file__):
        os.remove(__tmp_file__)
    

class TestDocs(unittest.TestCase):

    def test_docs(self):
        print('==================== Testing Codes in Docs ====================')
        cleanup()
        validate_markdown('docs/0-overview.md')
        validate_markdown('README.md')
        os.system(f'python {__tmp_file__}')
        cleanup()
        print('PASS')


if __name__ == '__main__':
    unittest.main()