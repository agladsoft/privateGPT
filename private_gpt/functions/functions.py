import requests


# Функция для интеграции в LLM через tools
tools = [
    {
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
    },
    {
        "type": "function",
        "function": {
            "name": "calculate",
            "description": "Performs a mathematical operation (addition, subtraction, multiplication, division) "
                           "on two numbers",
            "parameters": {
                "type": "object",
                "properties": {
                    "operation": {
                        "type": "string",
                        "description": "The mathematical operation to perform. "
                                       "Supported operations: add, subtract, multiply, divide",
                    },
                    "number_one": {
                        "type": "number",
                        "description": "The first number for the operation",
                    },
                    "number_two": {
                        "type": "number",
                        "description": "The second number for the operation",
                    },
                },
                "required": ["operation", "number_one", "number_two"],
            },
        },
    }
]


def get_current_weather(latitude, longitude):
    # Используем OpenWeatherMap API или другой API для получения текущей погоды
    api_key = "59bf8b2f47201a898a0612f31da81190"
    url = f"http://api.openweathermap.org/data/2.5/weather?lat={latitude}&lon={longitude}&appid={api_key}" \
          f"&units=metric&lang=ru"
    response = requests.get(url, timeout=60)
    if response.status_code != 200:
        return {"error": "Unable to fetch weather data"}
    weather_data = response.json()
    return f'На данный момент сейчас температура {weather_data["main"]["temp"]} градусов по Цельсию. ' \
           f'Погода - { weather_data["weather"][0]["description"]}. Локация - {weather_data["name"]}'


def calculate(operation, number_one, number_two):
    if operation == "add":
        return f"Ответ является {number_one + number_two}"
    elif operation == "subtract":
        return f"Ответ является {number_one - number_two}"
    elif operation == "multiply":
        return f"Ответ является {number_one * number_two}"
    elif operation == "divide":
        return f"Ответ является {number_one / number_two}"
    else:
        raise ValueError(f"Unknown operation {operation}")
