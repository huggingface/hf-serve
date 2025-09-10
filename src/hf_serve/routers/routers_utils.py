from typing import List

import magic


class FileValidator:
    def __init__(self, accepted_mimetypes: List[str], max_size: int = None):
        self.accepted_mimetypes = accepted_mimetypes
        self.max_size = max_size

    async def validate_file(self, file: bytes) -> List:
        errors = []

        file_size = len(file)
        if self.max_size and (file_size > self.max_size):
            errors.append(
                f"File size exceeded ({file_size:,} bytes). Maximum: {self.max_size:,} bytes."
            )

        mime_type = magic.from_buffer(file, mime=True)
        if mime_type not in self.accepted_mimetypes:
            errors.append(
                f"File MIME type not allowed for this task ({mime_type}). Allowed MIME types: {', '.join(self.accepted_mimetypes)}."
            )

        return errors
