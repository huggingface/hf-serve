
from typing import Dict, List

import magic


class DocumentValidator:
    def __init__(self, accepted_mimetypes: List[str], max_size: int = 10 * 1024 * 1024):
        self.accepted_mimetypes = accepted_mimetypes
        self.max_size = max_size

    async def validate_file(self, file: bytes) -> Dict:
        result = {"valid": True, "errors": []}

        file_size = len(file)
        if self.max_size and (file_size > self.max_size):
            result["valid"] = False
            result["errors"].append(
                f"File size exceeded ({file_size:,} bytes). Maximum: {self.max_size:,} bytes."
            )
        
        mime_type = magic.from_buffer(file, mime=True)
        if mime_type not in self.accepted_mimetypes:
            result["valid"] = False
            result["errors"].append(
                f"File MIME type not allowed for this task ({mime_type}). Allowed MIME types: {", ".join(self.accepted_mimetypes)}."
            )
        
        return result

