import os
from pathlib import Path
import sys



current_path = os.path.abspath(__file__)
root_path    = str(Path(current_path).parent.absolute())
sys.append(current_path)
