from typing import Annotated

from fastapi import File, Form

FileForm = Annotated[bytes, File(...)]

StrForm = Annotated[str, Form()]
IntForm = Annotated[int, Form()]
FloatForm = Annotated[float, Form()]
BoolForm = Annotated[bool, Form()]
