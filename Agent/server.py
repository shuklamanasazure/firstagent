# Create the MCP server, We will use two real APIs: Weather: Open-Meteo (free, no API key required). 
# Currency: Frankfurter API (exchange-rate data, no API key required).

from fastmcp import FastMCP
import httpx

mcp = FASTMCP("MCP Server", "0.1.0")

#Tool to get weather data from Open-Meteo API
@mcp.tool()
async def get_weather(location: str) -> dict:
    """Get weather data for a given location using the Open-Meteo API."""
    async with httpx.AsyncClient() as client:
        # Find city coordinates using Open-Meteo's geocoding API
        geocode_response = await client.get(f"https://api.open-meteo.com/v1/geocode?name={location}")
        geocode_data = geocode_response.json()
        latitude = geocode_data["results"][0]["latitude"]
        longitude = geocode_data["results"][0]["longitude"]
        if not latitude or not longitude:
            raise ValueError(f"Could not find coordinates for location: {location}")

        # Get current weather data using the coordinates

        response = await client.get(f"https://api.open-meteo.com/v1/forecast?latitude={latitude}&longitude={longitude}&current_weather=true")
        response.raise_for_status()
        
        return {
            "city": location,
            "country": geocode_data["results"][0]["country"],
            "latitude": latitude,
            "longitude": longitude,
            "weather": response.json()
        }

# Tool to get currency exchange rates from Frankfurter API
@mcp.tool()
async def get_exchange_rate(amount: float, base_currency: str, target_currency: str) -> dict:
    """Get exchange rate data for a given currency pair using the Frankfurter API."""

    if amount <= 0:
        raise ValueError("Amount must be greater than zero.")
    
    async with httpx.AsyncClient() as client:
        response = await client.get(f"https://api.frankfurter.app/latest?from={base_currency}&to={target_currency}")
        response.raise_for_status()
        data = response.json()
        return {
            "amount": amount,
            "base_currency": base_currency,
            "target_currency": target_currency,
            "exchange_rate": data["rates"][target_currency],
            "converted_amount": amount * data["rates"][target_currency],
            "date": data["date"]
        }

if __name__ == "__main__":
    # Run the MCP server on localhost:8000
    mcp.run(host="127.0.0.1", port=8000)

# We have exposed two Python functions as MCP tools using the @mcp.tool() decorator.

# FastMCP automatically generates the tool schemas from function signatures and docstrings.