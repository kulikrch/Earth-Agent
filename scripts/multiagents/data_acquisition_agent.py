"""
Data Acquisition Agent - Specialized ReAct agent for satellite data acquisition

This agent is responsible for finding and downloading satellite imagery from
Google Earth Engine when local data is not available. It uses intelligent retry
strategies to handle various failure scenarios.

Architecture:
- ReAct agent with EarthEngine MCP tools only
- Adaptive retry strategies (different collections, cloud thresholds, time windows)
- Natural language requirements → Downloaded GeoTIFF files
"""

import asyncio
import re
import json
import logging
import os
from typing import Dict, List, Any, Optional
from pathlib import Path
from typing_extensions import TypedDict
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage, BaseMessage
from .utils import extract_text_from_content, try_parse_json_with_required_key


class FileInfo(TypedDict, total=False):
    """Информация о скачанном файле"""
    path: str
    band: str
    date: str
    period_label: str
    semantic_role: str
    analysis_use: str


class ChannelInfo(TypedDict):
    """Информация о канале/полосе"""
    band: str
    description: str
    wavelength: Optional[str]


class ValidationResult(TypedDict):
    """Результат валидации скачанных файлов"""
    valid: bool
    existing_files: List[str]
    missing_files: List[str]
    total_files: int
    error_message: str


class AcquisitionResult(TypedDict):
    """Результат работы Data Acquisition Agent"""
    success: bool
    downloaded_files: List[FileInfo]
    channel_summary: Optional[List[ChannelInfo]]
    collection: Optional[str]
    reasoning: Optional[str]
    messages: List[BaseMessage]
    total_attempts: int
    tool_calls: List[Dict[str, Any]]
    agent_thoughts: List[str]
    error: Optional[str]

class DataAcquisitionAgent:
    """
    Специализированный ReAct агент для поиска и скачивания спутниковых данных.
    
    Использует только EarthEngine MCP tools для автоматического поиска снимков
    с умными retry стратегиями.
    """
    
    def __init__(self, llm: ChatOpenAI, earthengine_tools: List[Any], max_iterations: int = 20) -> None:
        """
        Инициализация Data Acquisition Agent.
        
        Args:
            llm: ChatOpenAI модель (рекомендуется gpt-4o для сложной retry логики)
            earthengine_tools: Список EarthEngine MCP tools (обёрнутых для sync если нужно)
            max_iterations: Максимальное количество ReAct итераций (по умолчанию 20 для retry стратегий)
        """
        self.llm: ChatOpenAI = llm
        self.earthengine_tools: List[Any] = earthengine_tools
        self.max_iterations: int = max_iterations
        self.logger: logging.Logger = logging.getLogger("DataAcquisitionAgent")
        
        with open('scripts/prompts/data_acquisition_agent.md', 'r', encoding='utf-8') as f:
            self.system_prompt: str = f.read()
        
        # Создаём ReAct агента
        from langgraph.prebuilt import create_react_agent
        self.agent = create_react_agent(self.llm, self.earthengine_tools)
    
    
    def _validate_downloaded_files(self, files: List[FileInfo]) -> ValidationResult:
        """
        Проверяет существование скачанных файлов по абсолютным путям.
        
        Args:
            files: Список файлов из ответа агента [{"path": "...", "band": "..."}, ...]
        
        Returns:
            ValidationResult с детальной информацией о валидации
        """
        existing_files: List[str] = []
        missing_files: List[str] = []
        
        self.logger.info(f"🔍 Validating {len(files)} downloaded files...")
        
        for file_info in files:
            file_path = file_info.get("path", "")
            band = file_info.get("band", "Unknown")
            
            if not file_path:
                self.logger.warning(f"⚠️ Empty path for band {band}")
                missing_files.append(f"[Empty path for band {band}]")
                continue
            
            if os.path.isfile(file_path):
                existing_files.append(file_path)
                self.logger.info(f"✅ Found: {file_path}")
            else:
                missing_files.append(file_path)
                self.logger.warning(f"❌ Missing: {file_path}")
        
        if missing_files:
            error_msg = f"Missing {len(missing_files)}/{len(files)} files: {', '.join(missing_files)}"
            self.logger.error(f"❌ Validation FAILED: {error_msg}")
            
            return ValidationResult(
                valid=False,
                existing_files=existing_files,
                missing_files=missing_files,
                total_files=len(files),
                error_message=error_msg
            )
        else:
            self.logger.info(f"✅ Validation SUCCESS: All {len(files)} files exist")
            return ValidationResult(
                valid=True,
                existing_files=existing_files,
                missing_files=[],
                total_files=len(files),
                error_message=""
            )

    @staticmethod
    def _normalize_text_path(path_text: str) -> str:
        path_text = (path_text or "").strip().strip('"').strip("'").strip("`")
        return path_text

    @staticmethod
    def _is_placeholder_path(path_text: str) -> bool:
        normalized = path_text.replace("\\", "/").lower()
        return (
            normalized.startswith("/full/path/")
            or normalized.startswith("c:/full/path/")
            or normalized.startswith("full/path/")
            or "/full/path/" in normalized
        )

    def _extract_saved_paths_from_messages(self, messages: List[BaseMessage]) -> List[str]:
        """
        Extract absolute file paths from tool results like:
        - Result saved at C:\\...\\file.tif
        - Result save at C:\\...\\file.tif
        """
        saved_paths: List[str] = []
        # Support both "saved" and historical typo "save".
        pattern = re.compile(r"Result sav(?:ed|e) at\s+([^\r\n]+?\.tif)", re.IGNORECASE)

        for message in messages:
            if not hasattr(message, "type") or message.type != "tool":
                continue

            raw_content = message.content
            candidates: List[str] = []

            if isinstance(raw_content, list):
                for item in raw_content:
                    if isinstance(item, str):
                        candidates.append(item)
                    else:
                        candidates.append(str(item))
            else:
                content_str = str(raw_content) if raw_content is not None else ""
                try:
                    parsed = json.loads(content_str)
                    if isinstance(parsed, list):
                        for item in parsed:
                            candidates.append(str(item))
                    else:
                        candidates.append(content_str)
                except Exception:
                    candidates.append(content_str)

            for text in candidates:
                for match in pattern.findall(text):
                    extracted = self._normalize_text_path(match)
                    if extracted:
                        saved_paths.append(extracted)

        # Keep existing only, deduplicate with stable order
        unique_existing: List[str] = []
        seen = set()
        for p in saved_paths:
            if p in seen:
                continue
            if os.path.isfile(p):
                seen.add(p)
                unique_existing.append(p)
        return unique_existing

    def _repair_file_paths(self, files: List[FileInfo], messages: List[BaseMessage]) -> List[FileInfo]:
        """
        Repair file paths returned by LLM using actual tool outputs.
        Helps recover from placeholder paths like /full/path/... and C:\\full\\path\\...
        """
        if not files:
            return files

        discovered_paths = self._extract_saved_paths_from_messages(messages)
        by_name: Dict[str, List[str]] = {}
        for file_path in discovered_paths:
            by_name.setdefault(Path(file_path).name, []).append(file_path)

        repaired: List[FileInfo] = []
        for file_info in files:
            path_value = self._normalize_text_path(str(file_info.get("path", "")))
            band_value = file_info.get("band", "Unknown")
            date_value = file_info.get("date")

            chosen_path = path_value
            path_exists = bool(path_value and os.path.isfile(path_value))
            is_placeholder = self._is_placeholder_path(path_value)
            is_relative = bool(path_value and not os.path.isabs(path_value))

            if (not path_exists) and path_value:
                fname = Path(path_value).name
                candidates = by_name.get(fname, [])
                if candidates:
                    # Prefer most recently modified candidate if multiple exist
                    candidates = sorted(
                        candidates,
                        key=lambda p: os.path.getmtime(p) if os.path.exists(p) else 0,
                        reverse=True,
                    )
                    chosen_path = candidates[0]
                    self.logger.info(
                        f"🔧 Repaired path for {fname}: '{path_value}' -> '{chosen_path}'"
                    )
                    path_exists = True

            if (is_placeholder or is_relative) and not path_exists:
                self.logger.warning(
                    f"⚠️ Unresolved {'placeholder' if is_placeholder else 'relative'} path: {path_value}"
                )

            repaired_item: Dict[str, Any] = dict(file_info)
            repaired_item["path"] = chosen_path
            repaired_item["band"] = band_value
            if date_value is not None:
                repaired_item["date"] = date_value
            repaired.append(repaired_item)  # type: ignore[arg-type]

        return repaired
    
    
    def _parse_final_response(self, messages: List[BaseMessage]) -> Optional[Dict[str, Any]]:
        """
        Парсит структурированный JSON ответ из финального сообщения агента.
        
        Args:
            messages: Список сообщений от ReAct агента
            
        Returns:
            Распарсенный JSON ответ или None
            
        Note:
            Агент должен вернуть JSON с информацией о скачанных файлах и метаданных.
        """
        for message in reversed(messages):
            if hasattr(message, 'type') and message.type == 'ai':
                # Извлекаем текст из content (совместимо с use_responses_api)
                if isinstance(message.content, list):
                    content = extract_text_from_content(message.content)
                else:
                    content = str(message.content) if message.content else ""
                
                if not content:
                    continue
                
                # Используем общий метод парсинга
                result = try_parse_json_with_required_key(content, "success")
                if result and result.get("success") is not None:
                    return result
        
        return None

    async def _request_file_redownload(
        self, 
        messages: List[BaseMessage], 
        missing_files: List[str], 
        validation_attempt: int
    ) -> Dict[str, Any]:
        """
        Запрашивает у агента повторное скачивание отсутствующих файлов.
        
        Args:
            messages: История сообщений предыдущего взаимодействия
            missing_files: Список путей к отсутствующим файлам
            validation_attempt: Номер попытки валидации (1-3)
        
        Returns:
            Dict с результатом повторного запроса
        """
        self.logger.warning(f"⚠️ File validation failed (attempt {validation_attempt}/3)")
        self.logger.info(f"🔄 Requesting redownload of {len(missing_files)} missing files...")
        
        # Формируем запрос на повторное скачивание
        redownload_prompt = f"""
ВНИМАНИЕ: Проверка скачанных файлов показала, что {len(missing_files)} файл(ов) отсутствует по указанным путям.

Отсутствующие файлы:
{chr(10).join(f'- {path}' for path in missing_files)}

Верни в массиве files в поле path абсолютные пути к файлам или повтори скачивание.

Верни структурированный JSON ответ согласно формату из системного промпта.
"""
        
        # Добавляем redownload запрос к существующим сообщениям (используем LangChain класс)
        extended_messages: List[BaseMessage] = list(messages) + [HumanMessage(content=redownload_prompt)]
        
        try:
            # Повторный вызов агента
            response = await asyncio.to_thread(
                self.agent.invoke,
                {"messages": extended_messages},
                {"recursion_limit": 30}  # Ограниченное количество итераций
            )
            
            new_messages = response.get("messages", [])
            
            # Парсим ответ
            parsed_response = self._parse_final_response(new_messages)
            
            if parsed_response:
                self.logger.info(f"✅ Got structured response after redownload request")
                self.logger.info(f"   Success: {parsed_response.get('success', False)}")
                return {
                    "parsed_response": parsed_response,
                    "messages": new_messages,
                    "redownload_used": True
                }
            else:
                self.logger.error("❌ Agent did not return structured JSON after redownload request")
                return {
                    "parsed_response": None,
                    "messages": new_messages,
                    "redownload_used": True
                }
        
        except Exception as e:
            self.logger.error(f"❌ Error during redownload request: {e}", exc_info=True)
            return {
                "parsed_response": None,
                "messages": messages,
                "redownload_used": False,
                "error": str(e)
            }
    
    async def _request_retry_confirmation(self, messages: List[BaseMessage]) -> Dict[str, Any]:
        """
        Запрашивает у агента подтверждение, что он испробовал все retry стратегии.
        
        Вызывается когда агент не вернул структурированный JSON ответ.
        Агент должен либо:
        1) Продолжить retry и вернуть успешный результат с JSON
        2) Подтвердить неудачу и вернуть {"success": false} с объяснением
        
        Args:
            messages: История сообщений предыдущего взаимодействия
        
        Returns:
            Dict с результатом повторного запроса
        """
        self.logger.warning("⚠️ Agent did not return structured JSON response")
        self.logger.info("🔄 Requesting retry confirmation from agent...")
        
        # Формируем уточняющий запрос
        clarification_prompt = f"""
ВНИМАНИЕ: Ты не вернул структурированный JSON ответ или вернул "success": false в предыдущем взаимодействии.

Пожалуйста, проверь:
1. Ты испробовал ВСЕ retry стратегии из промпта? (разные коллекции, cloud thresholds, временные окна, расширение bbox)
2. Если ты скачал файлы - верни валидный JSON ответ с информацией о них
3. Если после ВСЕХ попыток не удалось найти данные - верни JSON с "success": false

Верни структурированный JSON ответ согласно формату из системного промпта.
"""
        
        # Добавляем clarification к существующим сообщениям (используем LangChain класс)
        extended_messages: List[BaseMessage] = list(messages) + [HumanMessage(content=clarification_prompt)]
        
        try:
            # Повторный вызов агента
            response = await asyncio.to_thread(
                self.agent.invoke,
                {"messages": extended_messages},
                {"recursion_limit": 30}  # Ограниченное количество итераций для clarification
            )
            
            new_messages = response.get("messages", [])
            
            # Парсим ответ
            parsed_response = self._parse_final_response(new_messages)
            
            if parsed_response:
                self.logger.info(f"✅ Got structured response after clarification")
                self.logger.info(f"   Success: {parsed_response.get('success', False)}")
                return {
                    "parsed_response": parsed_response,
                    "messages": new_messages,
                    "clarification_used": True
                }
            else:
                self.logger.error("❌ Agent still did not return structured JSON after clarification")
                return {
                    "parsed_response": None,
                    "messages": new_messages,
                    "clarification_used": True
                }
        
        except Exception as e:
            self.logger.error(f"❌ Error during retry confirmation: {e}", exc_info=True)
            return {
                "parsed_response": None,
                "messages": messages,
                "clarification_used": False,
                "error": str(e)
            }

    def _log_message_details(self, messages: List[BaseMessage]) -> tuple[List[Dict[str, Any]], List[str]]:
        """
        Логирует детали взаимодействия агента и извлекает tool calls и мысли агента.
        
        Args:
            messages: Список сообщений от ReAct агента
            
        Returns:
            Кортеж (tool_calls, agent_thoughts)
        """
        tool_calls: List[Dict[str, Any]] = []
        agent_thoughts: List[str] = []
        
        for msg in messages:
            if not hasattr(msg, 'type'):
                continue
                
            if msg.type == 'ai':
                # Мысли агента (извлекаем текст из нового формата)
                if isinstance(msg.content, list):
                    content_text = extract_text_from_content(msg.content)
                else:
                    content_text = str(msg.content) if msg.content else ""
                
                if content_text and content_text.strip():
                    thought_preview = content_text[:200].replace('\n', ' ')
                    agent_thoughts.append(content_text)
                    self.logger.info(f"💭 Agent thinking: {thought_preview}...")
                
                # Tool calls
                if hasattr(msg, 'additional_kwargs') and 'tool_calls' in msg.additional_kwargs:
                    for tc in msg.additional_kwargs['tool_calls']:
                        tool_name = tc['function']['name']
                        tool_args_str = tc['function']['arguments']
                        
                        # Parse args для лучшего логирования
                        try:
                            tool_args = json.loads(tool_args_str) if isinstance(tool_args_str, str) else tool_args_str
                            args_preview = json.dumps(tool_args, ensure_ascii=False)[:150]
                        except:
                            args_preview = str(tool_args_str)[:150]
                        
                        tool_calls.append({"name": tool_name, "args": tool_args_str})
                        self.logger.info(f"🔧 Tool call: {tool_name}({args_preview}...)")
            
            elif msg.type == 'tool':
                # Tool results
                result_preview = str(msg.content)[:300] if msg.content else "empty"
                
                # Специальная обработка для search_images результатов
                if msg.name == 'search_images':
                    try:
                        result_data = json.loads(msg.content) if isinstance(msg.content, str) else msg.content
                        if isinstance(result_data, list):
                            self.logger.info(f"📊 Tool result ({msg.name}): Found {len(result_data)} images")
                            if len(result_data) > 0 and isinstance(result_data[0], dict):
                                self.logger.info(f"   First image: {result_data[0].get('id', 'N/A')}, cloud: {result_data[0].get('cloud_coverage', 'N/A')}")
                        else:
                            self.logger.info(f"📊 Tool result ({msg.name}): {result_preview}...")
                    except:
                        self.logger.info(f"📊 Tool result ({msg.name}): {result_preview}...")
                else:
                    self.logger.info(f"📊 Tool result ({msg.name}): {result_preview}...")
                
                # Логируем retry попытки
                if msg.name == 'search_images' and (not msg.content or msg.content == '[]' or msg.content == ''):
                    self.logger.warning(f"🔄 No images found - agent should retry with different parameters")
        
        return tool_calls, agent_thoughts
    
    async def search(self, requirements: str) -> AcquisitionResult:
        """
        Поиск и скачивание спутниковых данных через ReAct агента.
        
        Args:
            requirements: Описание требований к данным на естественном языке
                Пример:
                "Find and download satellite imagery for urban heat island analysis:
                - Location: bbox=[37.464, 55.568, 37.504, 55.608]
                - Period 1: summer 2018 (June-August)
                - Period 2: summer 2022 (June-August)
                - Required bands: thermal (B10, B11) for LST calculation
                - Save to directory: question3/"
        
        Returns:
            AcquisitionResult с результатами поиска и скачивания
        """
        self.logger.info(f"📡 Data Acquisition Agent: Starting search")
        self.logger.info(f"   Requirements preview: {requirements[:200]}...")
        
        # Формируем полный запрос
        full_query = f"Требования к данным:\n{requirements}"
        
        try:
            # Вызываем ReAct агента
            self.logger.info(f"🤖 Invoking ReAct agent (max_iterations={self.max_iterations})")
            
            # Используем LangChain классы сообщений вместо dict
            messages_input: List[BaseMessage] = [
                SystemMessage(content=self.system_prompt),
                HumanMessage(content=full_query)
            ]
            
            response = await asyncio.to_thread(
                self.agent.invoke,
                {"messages": messages_input},
                {"recursion_limit": self.max_iterations}
            )
            
            # Логируем детали взаимодействия
            messages = response.get("messages", [])
            tool_calls, agent_thoughts = self._log_message_details(messages)
            
            # Parse JSON response from agent's final message
            parsed_response = self._parse_final_response(messages)
            
            # Если агент вернул структурированный JSON ответ
            if parsed_response and parsed_response.get("success"):
                files = parsed_response.get("files", [])
                files = self._repair_file_paths(files, messages)
                collection = parsed_response.get("collection", "Unknown")
                reasoning = parsed_response.get("reasoning", "")
                channels_info = parsed_response.get("channels", [])
                
                self.logger.info(f"✅ Data acquisition successful: {len(files)} files downloaded")
                self.logger.info(f"📡 Collection used: {collection}")
                if reasoning:
                    self.logger.info(f"💡 Reasoning: {reasoning[:200]}...")
                
                # 🆕 ВАЛИДАЦИЯ ФАЙЛОВ (попытка 1 из 3)
                validation_result = self._validate_downloaded_files(files)
                
                if validation_result["valid"]:
                    # Все файлы существуют - успех!
                    return AcquisitionResult(
                        success=True,
                        downloaded_files=files,
                        channel_summary=channels_info,
                        collection=collection,
                        reasoning=reasoning,
                        messages=messages,
                        total_attempts=len(tool_calls),
                        tool_calls=tool_calls,
                        agent_thoughts=agent_thoughts,
                        error=None
                    )
                else:
                    # Файлы отсутствуют - retry loop (максимум 3 попытки)
                    max_validation_attempts = 3
                    current_messages = messages
                    
                    for validation_attempt in range(1, max_validation_attempts + 1):
                        self.logger.warning(f"🔄 File validation attempt {validation_attempt}/{max_validation_attempts}")
                        
                        # Запрашиваем повторное скачивание
                        redownload_result = await self._request_file_redownload(
                            current_messages,
                            validation_result["missing_files"],
                            validation_attempt
                        )
                        
                        current_messages = redownload_result.get("messages", current_messages)
                        retry_parsed_response = redownload_result.get("parsed_response")
                        
                        if not retry_parsed_response or not retry_parsed_response.get("success"):
                            # Агент не смог повторно скачать
                            self.logger.error(f"❌ Redownload failed on attempt {validation_attempt}")
                            if validation_attempt == max_validation_attempts or (retry_parsed_response and not retry_parsed_response.get("success")):
                                # Исчерпали все попытки или агент вернул false, а значит лучше уже не станет
                                return AcquisitionResult(
                                    success=False,
                                    downloaded_files=[],
                                    channel_summary=None,
                                    collection=None,
                                    reasoning=None,
                                    messages=current_messages,
                                    total_attempts=len(tool_calls),
                                    tool_calls=tool_calls,
                                    agent_thoughts=agent_thoughts,
                                    error=f"File validation failed after {max_validation_attempts} attempts: {validation_result['error_message']}"
                                )
                            continue  # Следующая попытка
                        
                        # Проверяем новые файлы
                        retry_files = retry_parsed_response.get("files", [])
                        retry_files = self._repair_file_paths(retry_files, current_messages)
                        retry_validation = self._validate_downloaded_files(retry_files)
                        
                        if retry_validation["valid"]:
                            # Успех после retry!
                            self.logger.info(f"✅ Files validated successfully after {validation_attempt} retry attempt(s)")
                            return AcquisitionResult(
                                success=True,
                                downloaded_files=retry_files,
                                channel_summary=retry_parsed_response.get("channels", []),
                                collection=retry_parsed_response.get("collection", collection),
                                reasoning=retry_parsed_response.get("reasoning", reasoning),
                                messages=current_messages,
                                total_attempts=len(tool_calls),
                                tool_calls=tool_calls,
                                agent_thoughts=agent_thoughts,
                                error=None
                            )
                        else:
                            # Все еще есть отсутствующие файлы
                            validation_result = retry_validation
                            if validation_attempt == max_validation_attempts:
                                # Исчерпали все попытки
                                return AcquisitionResult(
                                    success=False,
                                    downloaded_files=[],
                                    channel_summary=None,
                                    collection=None,
                                    reasoning=None,
                                    messages=current_messages,
                                    total_attempts=len(tool_calls),
                                    tool_calls=tool_calls,
                                    agent_thoughts=agent_thoughts,
                                    error=f"File validation failed after {max_validation_attempts} attempts: {retry_validation['error_message']}"
                                )
                    
                    # Не должно дойти сюда, но на всякий случай
                    return AcquisitionResult(
                        success=False,
                        downloaded_files=[],
                        channel_summary=None,
                        collection=None,
                        reasoning=None,
                        messages=current_messages,
                        total_attempts=len(tool_calls),
                        tool_calls=tool_calls,
                        agent_thoughts=agent_thoughts,
                        error=f"File validation failed: {validation_result['error_message']}"
                    )
            else:
                # Агент не вернул JSON - запрашиваем clarification
                clarification_result = await self._request_retry_confirmation(messages)
                
                # Используем сообщения из clarification
                messages = clarification_result.get("messages", messages)
                parsed_response = clarification_result.get("parsed_response")
                
                # Если после clarification получили структурированный JSON
                if parsed_response and parsed_response.get("success"):
                    files = parsed_response.get("files", [])
                    files = self._repair_file_paths(files, messages)
                    self.logger.info(f"✅ Fallback parsing successful: {len(files)} files found")
                    
                    collection = parsed_response.get("collection", "Unknown")
                    reasoning = parsed_response.get("reasoning", "")
                    channels_info = parsed_response.get("channels", [])
                    
                    # 🆕 ВАЛИДАЦИЯ ФАЙЛОВ (после clarification)
                    validation_result = self._validate_downloaded_files(files)
                    
                    if validation_result["valid"]:
                        # Все файлы существуют - успех!
                        return AcquisitionResult(
                            success=True,
                            downloaded_files=files,
                            channel_summary=channels_info,
                            collection=collection,
                            reasoning=reasoning,
                            messages=messages,
                            total_attempts=len(tool_calls),
                            tool_calls=tool_calls,
                            agent_thoughts=agent_thoughts,
                            error=None
                        )
                    else:
                        # Файлы отсутствуют - retry loop (максимум 3 попытки)
                        max_validation_attempts = 3
                        current_messages = messages
                        
                        for validation_attempt in range(1, max_validation_attempts + 1):
                            self.logger.warning(f"🔄 File validation attempt {validation_attempt}/{max_validation_attempts} (after clarification)")
                            
                            # Запрашиваем повторное скачивание
                            redownload_result = await self._request_file_redownload(
                                current_messages,
                                validation_result["missing_files"],
                                validation_attempt
                            )
                            
                            current_messages = redownload_result.get("messages", current_messages)
                            retry_parsed_response = redownload_result.get("parsed_response")
                            
                            if not retry_parsed_response or not retry_parsed_response.get("success"):
                                # Агент не смог повторно скачать
                                self.logger.error(f"❌ Redownload failed on attempt {validation_attempt}")
                                if validation_attempt == max_validation_attempts or (retry_parsed_response and not retry_parsed_response.get("success")):
                                    # Исчерпали все попытки или агент вернул false, а значит лучше уже не станет
                                    return AcquisitionResult(
                                        success=False,
                                        downloaded_files=[],
                                        channel_summary=None,
                                        collection=None,
                                        reasoning=None,
                                        messages=current_messages,
                                        total_attempts=len(tool_calls),
                                        tool_calls=tool_calls,
                                        agent_thoughts=agent_thoughts,
                                        error=f"File validation failed after {max_validation_attempts} attempts: {validation_result['error_message']}"
                                    )
                                continue  # Следующая попытка
                            
                            # Проверяем новые файлы
                            retry_files = retry_parsed_response.get("files", [])
                            retry_files = self._repair_file_paths(retry_files, current_messages)
                            retry_validation = self._validate_downloaded_files(retry_files)
                            
                            if retry_validation["valid"]:
                                # Успех после retry!
                                self.logger.info(f"✅ Files validated successfully after {validation_attempt} retry attempt(s)")
                                return AcquisitionResult(
                                    success=True,
                                    downloaded_files=retry_files,
                                    channel_summary=retry_parsed_response.get("channels", []),
                                    collection=retry_parsed_response.get("collection", collection),
                                    reasoning=retry_parsed_response.get("reasoning", reasoning),
                                    messages=current_messages,
                                    total_attempts=len(tool_calls),
                                    tool_calls=tool_calls,
                                    agent_thoughts=agent_thoughts,
                                    error=None
                                )
                            else:
                                # Все еще есть отсутствующие файлы
                                validation_result = retry_validation
                                if validation_attempt == max_validation_attempts:
                                    # Исчерпали все попытки
                                    return AcquisitionResult(
                                        success=False,
                                        downloaded_files=[],
                                        channel_summary=None,
                                        collection=None,
                                        reasoning=None,
                                        messages=current_messages,
                                        total_attempts=len(tool_calls),
                                        tool_calls=tool_calls,
                                        agent_thoughts=agent_thoughts,
                                        error=f"File validation failed after {max_validation_attempts} attempts: {retry_validation['error_message']}"
                                    )
                        
                        # Не должно дойти сюда, но на всякий случай
                        return AcquisitionResult(
                            success=False,
                            downloaded_files=[],
                            channel_summary=None,
                            collection=None,
                            reasoning=None,
                            messages=current_messages,
                            total_attempts=len(tool_calls),
                            tool_calls=tool_calls,
                            agent_thoughts=agent_thoughts,
                            error=f"File validation failed: {validation_result['error_message']}"
                        )
                else:
                    self.logger.warning(f"⚠️ Data acquisition failed: no files downloaded")
                    self.logger.warning(f"   Tool calls made: {[tc['name'] for tc in tool_calls]}")
                    
                    # Try to extract error reason from last AI message
                    error_reason = "No files were downloaded after all attempts"
                    for msg in reversed(messages):
                        if hasattr(msg, 'type') and msg.type == 'ai':
                            if isinstance(msg.content, list):
                                content_text = extract_text_from_content(msg.content)
                            else:
                                content_text = str(msg.content) if msg.content else ""
                            if content_text:
                                # Last agent message might contain explanation
                                if any(word in content_text.lower() for word in ['failed', 'error', 'unable', 'no images', 'not found']):
                                    error_reason = content_text[:500]
                                    break
                    
                    return AcquisitionResult(
                        success=False,
                        downloaded_files=[],
                        channel_summary=None,
                        collection=None,
                        reasoning=None,
                        messages=messages,
                        total_attempts=len(tool_calls),
                        tool_calls=tool_calls,
                        agent_thoughts=agent_thoughts,
                        error=error_reason
                    )
        
        except Exception as e:
            self.logger.error(f"❌ Error in Data Acquisition Agent: {e}", exc_info=True)
            return AcquisitionResult(
                success=False,
                downloaded_files=[],
                channel_summary=None,
                collection=None,
                reasoning=None,
                messages=[],
                total_attempts=0,
                tool_calls=[],
                agent_thoughts=[],
                error=str(e)
            )
