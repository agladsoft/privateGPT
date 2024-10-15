import requests


# Функция для интеграции в LLM через tools
weather = {
    "type": "function",
    "function": {
        "name": "get_current_weather",
        "description": "Get the current weather in a given latitude and longitude",
        "parameters": {
            "type": "object",
            "properties": {
                "latitude": {
                    "type": "number",
                    "description": "The latitude of a place",
                },
                "longitude": {
                    "type": "number",
                    "description": "The longitude of a place",
                },
            },
            "required": ["latitude", "longitude"],
        },
    },
}


def get_current_weather(latitude, longitude):
    # Используем OpenWeatherMap API или другой API для получения текущей погоды
    api_key = "59bf8b2f47201a898a0612f31da81190"
    url = f"http://api.openweathermap.org/data/2.5/weather?lat={latitude}&lon={longitude}&appid={api_key}&units=metric"

    response = requests.get(url, timeout=60)
    if response.status_code != 200:
        return {"error": "Unable to fetch weather data"}
    weather_data = response.json()
    return f'На данный момент сейчас температура {weather_data["main"]["temp"]} градусов по Цельсию. ' \
           f'Погода - { weather_data["weather"][0]["description"]}. Локация - {weather_data["name"]}'
