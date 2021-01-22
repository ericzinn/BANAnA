# pathTest.py

from pathlib import Path
basePath = "H:\\My Drive\\20200930 PS All Neut Data\\Output\\HS765"
p = Path(basePath + "/temp/").mkdir(parents=True, exist_ok=True)
