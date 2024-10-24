import os
from pathlib import Path
from private_gpt.constants import PROJECT_ROOT_PATH

FAVICON_PATH = 'https://i.ibb.co/DGGPZBG/logo.png'

THIS_DIRECTORY_RELATIVE = Path(__file__).parent.relative_to(PROJECT_ROOT_PATH)
# Should be "private_gpt/ui/avatar-bot.ico"
IMAGES = os.path.join(THIS_DIRECTORY_RELATIVE, "images")
if not os.path.exists(IMAGES):
    os.mkdir(IMAGES)
AVATAR_USER = f"{IMAGES}/icons8-человек-96.png"
AVATAR_BOT = f"{IMAGES}/icons8-bot-96.png"
LOGIN_ICON = f"{IMAGES}/login.png"
LOGOUT_ICON = f"{IMAGES}/logout.png"
MESSAGE_LOGIN = "Введите логин и пароль, чтобы войти"

QRCODE_PATH = f"{IMAGES}/qrcode.png"
