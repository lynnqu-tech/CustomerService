from pydantic import BaseModel, Field


class ErrorResponse(BaseModel):
    code: str = Field(description="Stable machine-readable error code")
    message: str = Field(description="Human-readable error summary")
    detail: str = Field(description="Troubleshooting detail")
