import os
import re

import aiofiles
from fastapi import Depends, FastAPI, HTTPException
from pydantic import BaseModel, Field

from src.type_definitions.constraint_types import SafeCodeString

app = FastAPI()

# Base directory for all generated files
BASE_DIRECTORY = os.path.join(os.getcwd(), "generated_files")
os.makedirs(BASE_DIRECTORY, exist_ok=True)


# Pydantic model for the file-generation request.
class FileRequest(BaseModel):
    # Tenant-specific filename: only allow alphanumerics, underscores, dashes, and must end with ".py"
    filename: str = Field(pattern=r"^[A-Za-z0-9_\-]+\.py$")
    code: SafeCodeString


# Dependency that returns the current tenant id.
# In a real application, this could extract the tenant id from the JWT or session.
def get_current_tenant() -> str:
    # For demonstration, we return a fixed tenant id.
    # Replace this with your actual logic to extract the tenant (or user) identifier.
    return "tenant_123"


def sanitize_code(code: str) -> str:
    # Basic sanitation: you could check for forbidden expressions or other security rules.
    forbidden_patterns = [r"import\s+os", r"import\s+sys", r"__import__"]
    for pattern in forbidden_patterns:
        if re.search(pattern, code):
            raise HTTPException(
                status_code=400, detail="File content contains forbidden expressions"
            )
    return code


@app.post("/create_file", status_code=201)
async def create_file(file_request: FileRequest, tenant_id: str = Depends(get_current_tenant)):
    # Sanitize code (the filename was already validated by the model)
    safe_code = sanitize_code(file_request.code)

    # Create a directory for the tenant if it doesn't exist
    tenant_directory = os.path.join(BASE_DIRECTORY, tenant_id)
    os.makedirs(tenant_directory, exist_ok=True)

    # Construct the complete file path
    file_path = os.path.join(tenant_directory, file_request.filename)

    try:
        async with aiofiles.open(file_path, "w") as f:
            await f.write(safe_code)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error writing file: {e!s}") from e

    return {
        "message": f"File '{file_request.filename}' created successfully for tenant '{tenant_id}'",
        "path": file_path,
    }


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("api_prefect_create_file:app", host="0.0.0.0", port=8887, reload=True)  # noqa: S104
