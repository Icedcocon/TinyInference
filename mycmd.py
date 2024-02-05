from xinference_demo import local

import re
import sys
# from xinference.deploy.cmdline import local
if __name__ == '__main__':
    sys.argv[0] = re.sub(r'(-script\.pyw|\.exe)?$', '', sys.argv[0])
    print(f"Command line parameters are {sys.argv}")
    sys.exit(local())
