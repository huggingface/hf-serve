from typing import List, Optional

import magic


class FileValidator:
    def __init__(self, accepted_mimetypes: List[str], max_size: Optional[int] = None):
        self.accepted_mimetypes = accepted_mimetypes
        self.max_size = max_size

    def __call__(self, file: bytes) -> None:
        mime_type = magic.from_buffer(file, mime=True)
        if not any(mime_type.startswith(m.split("/*")[0]) for m in self.accepted_mimetypes):
            raise ValueError(
                f"File MIME type {mime_type} not allowed for this task. Allowed MIME types: {', '.join(self.accepted_mimetypes)}."
            )

        file_size = len(file)
        if self.max_size and (file_size > self.max_size):
            raise ValueError(f"File size exceeded ({file_size:,} bytes). Maximum: {self.max_size:,} bytes.")
