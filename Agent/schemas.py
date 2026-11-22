from typing import Literal, Optional
from pydantic import BaseModel, Field, ConfigDict, model_validator

# Common strict configuration.
class StrictSchema(BaseModel):
    model_config = ConfigDict(
        extra="forbid", 
        str_strip_whitespace=True
    )

# 1. INPUT SCHEMAS FOR MCP TOOLS

class GetWeatherInput(StrictSchema):
    location: str = Field(
        min_length=2,
        max_length=100, 
        description="The location for which to get weather"
        )

class GetExchangeRateInput(StrictSchema):
    amount: float = Field(
        gt=0,
        description="The amount to convert"
    )
    base_currency: str = Field(
        min_length=3,
        max_length=3,
        description="The base currency code (e.g., USD)"
    )
    target_currency: str = Field(
        min_length=3,
        max_length=3,
        description="The target currency code (e.g., EUR)"
    )

# 2. OUTPUT SCHEMAS FROM MCP TOOLS

class WeatherData(StrictSchema):
    temperature: float = Field(
        description="Current temperature in Celsius"
    )
    windspeed: float = Field(
        description="Current wind speed in km/h"
    )
    weathercode: int = Field(
        description="Weather code representing the current weather condition"
    )
    location: str = Field(
        description="The location for which the weather data is provided"
    )

class CurrencyExchangeData(StrictSchema):
    amount: float = Field(
        description="The amount to convert"
    )
    base_currency: str = Field(
        description="The base currency code (e.g., USD)"
    )
    target_currency: str = Field(
        description="The target currency code (e.g., EUR)"
    )
    exchange_rate: float = Field(
        description="The exchange rate from base_currency to target_currency"
    )
    converted_amount: float = Field(
        description="The converted amount in target_currency"
    )
    date: str = Field(
        description="The date of the exchange rate data"
    )

# 3. INPUT SCHEMA FOR EACH LLM ITERATION

class ToolObservation(StrictSchema):
    tool_name: str = Field(
        description="The name of the tool that was executed"
    )
    success: bool = Field(
        description="Indicates whether the tool execution was successful"
    )
    data: dict = Field(default_factory=dict, description="The data returned by the tool execution")
    error: Optional[str] = Field(default=None, description="Error message if the tool execution failed")

class LLMIterationInput(StrictSchema):
    user_input: str = Field(
        description="The input provided by the user for this iteration"
    )
    iteration_number: int = Field(
        ge=1,
        le=3,
        description="The iteration number of the LLM interaction"
    )
    max_iterations: int = 3
    tool_observations: list[ToolObservation] = Field(
        default_factory=list,
        description="A list of observations from tools executed in this iteration"
    )

# 4. STRUCTURED OUTPUT SCHEMA FROM LLM

class AgentStep(StrictSchema):
    action: Literal["call_tool", "final"]

    tool_name: Optional[
        Literal["weather_tool", "currency_tool"]
    ] = None

    arguments: Optional[dict] = None

    answer: Optional[str] = None

    @model_validator(mode="after")
    def validate_action_fields(self):

        # A tool action must have a tool name and arguments.
        if self.action == "call_tool":
            if not self.tool_name or self.arguments is None:
                raise ValueError(
                    "Tool action requires tool_name and arguments"
                )

            if self.answer is not None:
                raise ValueError(
                    "Tool action must not contain a final answer"
                )

        # A final action must contain an answer.
        if self.action == "final":
            if not self.answer:
                raise ValueError(
                    "Final action requires an answer"
                )

            if self.tool_name is not None or self.arguments is not None:
                raise ValueError(
                    "Final action must not contain tool arguments"
                )

        return self


# --------------------------------------------------
# 5. FINAL OUTPUT SCHEMA
# --------------------------------------------------

class AgentResponse(StrictSchema):
    status: Literal["success", "partial", "failed"]
    answer: str
    request_id: str
    iterations_used: int = Field(ge=1, le=3)
    observations: list[ToolObservation]
    used_tools: list[str]