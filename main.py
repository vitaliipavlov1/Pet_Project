from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
import requests
from openai import OpenAI
import os
import torch
from dotenv import load_dotenv
from unet_generator_chatbot.unet_generator import UNetGenerator
from patch_discriminator_chatbot import PatchDiscriminator
load_dotenv()


client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
ACCESS_TOKEN = os.getenv("ACCESS_TOKEN")
PHONE_NUMBER_ID = os.getenv("PHONE_NUMBER_ID")
VERIFY_TOKEN = os.getenv("VERIFY_TOKEN")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model_gen = UNetGenerator().eval().to(device)
model_dis = PatchDiscriminator().eval().to(device)

if not os.path.exists('model_gen.tar'):
    raise FileNotFoundError("Файл model_gen.tar не найден")

st_gen = torch.load('model_gen.tar', weights_only=True)
model_gen.load_state_dict(st_gen)

if not os.path.exists('model_dis.tar'):
    raise FileNotFoundError("Файл model_dis.tar не найден")

st_dis = torch.load('model_dis.tar', weights_only=True)
model_dis.load_state_dict(st_dis)

app = FastAPI()

# ✅ Системная инструкция
SYSTEM_PROMPT = '''Ты — чат-бот на испанском языке в whatsapp частной компании по разработке ПО. Все твои ответы должны быть только на испанском языке. Отвечай только на сообщения, связанные с этой компанией или ее услугами.
Любая нецензурная/нелегальная лексика недопустима ни со стороны пользователя, ни со стороны чат-бота.
В случае сообщения, не связанного с этой компанией или ее услугами выводи на испанском языке: Уважаемый пользователь, это - чат-бот компании {name_of_the_company}, если вы хотели бы получить информацию о контактах компании, графике работы или ее услугах, пожалуйста, укажите это в вашем сообщении.
В случае сообщения с нецензурной/нелегальной лексикой выводи сообщение на испанском языке: В соответствии с политикой компании любая нецензурная/нелегальная лексика полностью запрещена.'''


@app.get("/webhook")
async def verify_webhook(request: Request):
    params = dict(request.query_params)
    if params.get("hub.mode") == "subscribe" and params.get("hub.verify_token") == VERIFY_TOKEN:
        return JSONResponse(content=int(params.get("hub.challenge")))
    return JSONResponse(status_code=403, content={"error": "Verification failed"})

@app.post("/webhook")
async def receive_whatsapp_message(request: Request):
    body = await request.json()
    print("Получено:", body)

    try:
        messages = body["entry"][0]["changes"][0]["value"].get("messages")
        if messages:
            msg = messages[0]
            from_number = msg["from"]
            msg_type = msg['type']

            if msg_type == "text":
                text = msg["text"]["body"]
                reply_text = generate_chatgpt_reply(text)
                send_whatsapp_message(from_number, text=reply_text)

            elif msg_type == "image":

                media_id = msg["image"]["id"]

                media_url = get_image_url(media_id)

                image_bytes = download_image(media_url)

                real_img = image_bytes_to_tensor(image_bytes)  # Можно временно сохранить изображение

                img_gen = model_gen(real_img)
                dis_out = model_dis(img_gen)

                heatmap_gen = create_heatmap_image(img_gen)
                heatmap_dis = create_heatmap_image(dis_out)

                media_id_gen = upload_image_to_whatsapp(heatmap_gen)
                media_id_dis = upload_image_to_whatsapp(heatmap_dis)

                send_whatsapp_message(from_number, media_id=media_id_gen, caption='gen_generated')
                send_whatsapp_message(from_number, media_id=media_id_dis, caption='dis_generated')

            else:
                reply_text = f"Тип сообщения '{msg_type}' пока не поддерживается."
                send_whatsapp_message(from_number, reply_text)

    except Exception as e:
        print("Ошибка:", e)

    return {"status": "ok"}

def generate_chatgpt_reply(user_message: str) -> str:
    try:
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_message}
            ],
            temperature=0.2
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        print('Ошибка при обращении к ИИ.', e)
        return "Ошибка при обращении к ИИ. Попробуйте позже."


def send_whatsapp_message(to_number: str, text: Optional[str], media_id: Optional[str] = None, caption: Optional[str] = None):

    url = f"https://graph.facebook.com/v18.0/{PHONE_NUMBER_ID}/messages"
    headers = {
        "Authorization": f"Bearer {ACCESS_TOKEN}",
        "Content-Type": "application/json"
    }
    if text:
        payload = {
            "messaging_product": "whatsapp",
            "to": to_number,
            "type": "text",
            "text": {"body": text}
        }
    elif media_id:
        payload = {
            "messaging_product": "whatsapp",
            "to": to_number,
            "type": "image",
            "image": {
            "id": media_id,
            "caption": caption
        }
    }


    response = requests.post(url, json=payload, headers=headers)
    print("Ответ от WhatsApp API:", response.json())




def get_image_url(media_id: str) -> str:
    url = f"https://graph.facebook.com/v18.0/{media_id}"
    headers = {"Authorization": f"Bearer {WHATSAPP_TOKEN}"}
    response = requests.get(url, headers=headers)
    response.raise_for_status()
    return response.json()["url"]


def download_image(media_url: str) -> bytes:
    headers = {"Authorization": f"Bearer {WHATSAPP_TOKEN}"}
    response = requests.get(media_url, headers=headers)
    response.raise_for_status()
    return response.content   # bytes


def image_bytes_to_tensor(image_bytes: bytes) -> torch.Tensor:
    # Открытие изображения из байтов
    image = Image.open(BytesIO(image_bytes)).convert("RGB")

    # Преобразования: изменение размера, нормализация и преобразование в тензор
    transform = transforms.Compose([
        transforms.Resize((224, 224)),       # Например, под ResNet
        transforms.ToTensor(),               # Преобразует в [C x H x W], значения от 0 до 1
        transforms.Normalize(                # Нормализация (можно свои значения)
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])

    tensor = transform(image)  # [3, 224, 224]
    return tensor.unsqueeze(0) # [1, 3, 224, 224] — батч из одного изображения


def create_heatmap_image(tensor: torch.Tensor, title: str = "Heatmap anomaly detection") -> bytes:
    # Если это torch.Tensor, приводим к numpy
    if isinstance(tensor, torch.Tensor):
        tensor = tensor.detach().cpu().squeeze().numpy()

    # Создаем изображение
    fig, ax = plt.subplots()
    heatmap = ax.imshow(tensor, cmap='viridis')
    ax.set_title(title)
    plt.colorbar(heatmap)

    # Сохраняем в байтовый поток
    buf = BytesIO()
    plt.savefig(buf, format='JPEG')
    plt.close(fig)
    buf.seek(0)
    return buf.read()


def upload_image_to_whatsapp(image_bytes: bytes) -> str:
    upload_url = f"https://graph.facebook.com/v18.0/{PHONE_NUMBER_ID}/media"
    headers = {
        "Authorization": f"Bearer {ACCESS_TOKEN}"
    }
    files = {
        'file': ('image.jpg', image_bytes, 'image/jpeg')
    }
    data = {
        'messaging_product': 'whatsapp'
    }
    response = requests.post(upload_url, headers=headers, files=files, data=data)
    response.raise_for_status()
    media_id = response.json()['id']
    return media_id




