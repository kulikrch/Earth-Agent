import os
import httpx
from dotenv import load_dotenv
import json
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Optional, Tuple
import openai
from openai.pagination import SyncCursorPage
from openai.types.responses import ResponseInputItem, ResponseItem, ResponsePrompt
from openai.types.responses.response import Conversation
from openai.types.responses.response_output_item import ResponseOutputItem

load_dotenv()


class ResponsesAPIClient:
    def __init__(self, api_key: str, proxy: str | None = None):
        """
        Инициализация клиента Responses API

        Args:
            api_key: OpenAI API ключ
            proxy: URL прокси (например: http://proxy.example.com:8080)
        """
        self.api_key = api_key
        self.base_url = "https://api.openai.com/v1"

        # Конфигурация прокси
        self.client = openai.OpenAI(
            api_key=api_key,
            http_client=httpx.Client(proxy=proxy if proxy else None, timeout=30.0)
        )

    def get_response_history(self, response_id: str) -> Tuple[Optional[List[ResponseOutputItem]], Optional[str], SyncCursorPage[ResponseItem] | None]:
        """Получить историю по response_id"""
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

        try:
            response = self.client.responses.retrieve(response_id, stream=False, include=["message.input_image.image_url"])

            input = self.client.responses.input_items.list(response_id)
            return response.output, response.previous_response_id, input

        except httpx.HTTPError as e:
            print(f"Ошибка HTTP при получении {response_id}: {e}")
            return None, None, None
    def get_full_chat_chain(self, response_id: str) -> List[Dict[str, str]] | None:
        """
        Получить полную цепочку чата, идя по previous_response_id

        Args:
            response_id: ID response, с которого начинаем

        Returns:
            Список всех responses в порядке от первого к последнему
        """
        chain: List[Dict[str, str]] = []
        current_id = response_id
        visited = set()

        print(f"Построение цепочки чата начиная с {response_id}...\n")

        fisrt_input: SyncCursorPage[ResponseItem] | None = None

        # Сначала идём в конец цепочки (где нет previous_response_id)
        while current_id and current_id not in visited:
            visited.add(current_id)
            response_data, previos_id, input = self.get_response_history(current_id)
            current_id = previos_id

            if response_data is None:
                return None
            
            if fisrt_input is None and input:
                fisrt_input = input

            for message in response_data:
                if message.type == 'message':
                    for c in message.content:
                        if c.type == "refusal":
                            chain.append({ "role": "assistant", "refusal": c.refusal })
                        if c.type == "output_text":
                            chain.append({ "role": "assistant", "content": c.text })
                if message.type == "function_call":
                    if fisrt_input:
                        for i in fisrt_input:
                            if i.type == "function_call_output" and i.call_id == message.call_id:
                                chain.append({ "role": "function_call_output", "function_call_output": i.output, "call_id": message.call_id })
                    chain.append({ "role": "assistant", "function_call": message.arguments, "call_id": message.call_id })

            if previos_id is None and input:
                for i in input:
                    if i.type == "message" and i.role == "system":
                        for c in i.content:
                            if c.type == "input_text":
                                chain.append({ "role": i.role, "content": c.text })

                    if i.type == "message" and i.role == "user":
                        for c in i.content:
                            if c.type == "input_text":
                                chain.append({ "role": i.role, "content": c.text })

        chain.reverse()

        return chain

    def save_to_json(
        self,
        initial_response_id: str,
        chain: List[Dict[str, str]],
        output_path: str | None = None,
    ) -> str | None:
        """
        Сохранить полную историю чата в JSON файл

        Args:
            initial_response_id: Начальный ID response
            chain: Цепочка responses (если None, получит автоматически)
            output_path: Путь для сохранения (опционально)

        Returns:
            Путь к созданному файлу
        """

        # Подготовить данные для сохранения
        data = {
            "initial_response_id": initial_response_id,
            "exported_at": datetime.now().isoformat(),
            "total_events": len(chain),
            "conversation": chain,
        }

        # Определить путь для сохранения
        if output_path is None:
            output_dir = Path("responses_data")
            output_dir.mkdir(exist_ok=True)

            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            safe_id = initial_response_id.replace("/", "_").replace("\\", "_")
            output_path = output_dir / f"chat_history_{safe_id}_{timestamp}.json" # type: ignore
        else:
            output_path = Path(output_path) # type: ignore
            output_path.parent.mkdir(parents=True, exist_ok=True) # type: ignore

        # Сохранить в JSON
        try:
            with open(output_path, "w", encoding="utf-8") as f: # type: ignore
                json.dump(data, f, indent=2, ensure_ascii=False)

            file_size = output_path.stat().st_size # type: ignore
            print(f"\n✓ Файл успешно сохранён: {output_path}")
            print(f"  Размер: {file_size:,} байт")
            print(f"  Всего событий: {len(chain)}")

            return str(output_path)

        except IOError as e:
            print(f"✗ Ошибка при сохранении файла: {e}")
            return None

    def close(self):
        """Закрыть соединение"""
        self.client.close()


# Использование
if __name__ == "__main__":
    api_key = os.getenv("OPENAI_API_KEY")

    if api_key is None:
        raise ValueError("Не указан API ключ OpenAI")

    proxy = os.getenv("PROXY_URL", None)

    client = ResponsesAPIClient(api_key, proxy)

    response_id = "resp_0136dbd0d3c29af300698e062cd0fc819796b7a23711a16485"

    try:
        # Получить полную цепочку чата
        chain = client.get_full_chat_chain(response_id)

        if chain:
            client.save_to_json(response_id, chain)
    finally:
        client.close()
