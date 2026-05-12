
# ============================================================================
# MULTI-AGENT ARCHITECTURE: Location Agent
# ============================================================================

# Multi-agent architecture imports
import asyncio
import json
import logging
from typing import Dict, List, Optional, TypedDict, Annotated, Any, Sequence
from langchain_core.messages import HumanMessage, SystemMessage, BaseMessage
import re
from langchain_openai import ChatOpenAI
from .utils import extract_text_from_content, try_parse_json_with_required_key



class LocationState(TypedDict):
    """Состояние Location Agent"""
    query: str
    location_found: bool
    coordinates: Optional[Dict[str, Any]]
    search_attempts: List[Dict[str, Any]]
    messages: Annotated[List[BaseMessage], "Messages"]
    retry_count: int
    max_retries: int


class CoordinatesResult(TypedDict):
    """Результат поиска координат"""
    found: bool
    lat: float
    lon: float
    bbox: List[float]
    display_name: str
    osm_type: str
    osm_id: str


class SearchResult(TypedDict):
    """Результат работы Location Agent"""
    found: bool
    coordinates: Optional[CoordinatesResult]
    messages: List[BaseMessage]
    total_attempts: int
    tool_calls: List[Dict[str, Any]]
    agent_thoughts: List[str]
    error: Optional[str]


class LocationAgent:
    """
    Специализированный ReAct агент для поиска координат.
    
    Использует LangGraph ReAct для автоматического поиска координат через OSM tools.
    Агент сам видит свою историю взаимодействия и адаптирует запросы.
    """
    
    def __init__(self, llm: ChatOpenAI, osm_tools: List[Any], max_iterations: int = 10) -> None:
        """
        Инициализация Location Agent.
        
        Args:
            llm: ChatOpenAI модель для ReAct агента
            osm_tools: Список OSM MCP tools (обёрнутых для sync если нужно)
            max_iterations: Максимальное количество ReAct итераций
        """
        self.llm: ChatOpenAI = llm
        self.osm_tools: List[Any] = osm_tools
        self.max_iterations: int = max_iterations
        self.logger: logging.Logger = logging.getLogger("LocationAgent")
        
        with open('scripts/prompts/location_agent.md', 'r', encoding='utf-8') as f:
            self.system_prompt: str = f.read()
        
        # Создаём ReAct агента
        from langgraph.prebuilt import create_react_agent
        self.agent = create_react_agent(self.llm, self.osm_tools)
    
    
    def _normalize_coordinates(self, data: Dict[str, Any]) -> CoordinatesResult:
        """
        Валидирует и приводит типы данных ответа модели к нужному формату.
        
        Args:
            data: Сырые данные координат от модели
            
        Returns:
            Нормализованный результат с координатами
            
        Raises:
            ValueError: Если формат bbox некорректный
            
        Note:
            Модель должна вернуть bbox согласно промпту
            Формат: [min_lon, min_lat, max_lon, max_lat]
        """
        bbox = data.get("bbox", [])
        if not isinstance(bbox, list) or len(bbox) != 4:
            raise ValueError(f"Invalid bbox format: {bbox}. Expected list of 4 floats.")
        
        return CoordinatesResult(
            found=True,
            lat=float(data["lat"]),
            lon=float(data["lon"]),
            bbox=[float(x) for x in bbox],
            display_name=data.get("display_name", ""),
            osm_type=data.get("osm_type", ""),
            osm_id=str(data.get("osm_id", ""))
        )
    
    
    def _parse_coordinates_from_response(self, response: Dict[str, Any]) -> Optional[CoordinatesResult]:
        """
        Извлекает координаты из финального ответа модели.
        
        Args:
            response: Ответ от ReAct агента
            
        Returns:
            Нормализованные координаты или None
            
        Note:
            Модель должна вернуть JSON в формате из промпта location_agent.md.
            Мы НЕ парсим tool results - это ответственность модели.
        """
        messages = response.get("messages", [])
        
        # Ищем последнее AI message с финальным ответом
        for message in reversed(messages):
            if hasattr(message, 'type') and message.type == 'ai':
                # Извлекаем текст из content (совместимо с use_responses_api)
                if isinstance(message.content, list):
                    content = extract_text_from_content(message.content)
                else:
                    content = str(message.content) if message.content else ""
                
                if not content:
                    continue
                
                # Пытаемся распарсить JSON с ключом "found"
                result = try_parse_json_with_required_key(content, "found")
                if result and result.get("found") and "lat" in result and "lon" in result:
                    return self._normalize_coordinates(result)
        
        # Если не нашли - модель не выполнила инструкции из промпта
        return None
    
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
                    agent_thoughts.append(content_text[:200])
                    self.logger.info(f"💭 Agent thinking: {content_text[:200]}...")
                
                # Tool calls
                if hasattr(msg, 'additional_kwargs') and 'tool_calls' in msg.additional_kwargs:
                    for tc in msg.additional_kwargs['tool_calls']:
                        tool_name = tc['function']['name']
                        tool_args = tc['function']['arguments']
                        tool_calls.append({"name": tool_name, "args": tool_args})
                        self.logger.info(f"🔧 Tool call: {tool_name}({tool_args})")
            
            elif msg.type == 'tool':
                # Результаты tool (могут быть в обычном формате)
                result_preview = str(msg.content)[:200] if msg.content else "empty"
                self.logger.info(f"📊 Tool result ({msg.name}): {result_preview}...")
        
        return tool_calls, agent_thoughts
    
    def _log_search_result(self, coordinates: Optional[CoordinatesResult], tool_calls: List[Dict[str, Any]]) -> None:
        """
        Логирует результат поиска координат.
        
        Args:
            coordinates: Найденные координаты или None
            tool_calls: Список вызовов инструментов
        """
        if coordinates and coordinates.get("found"):
            self.logger.info(f"✅ Location found: lat={coordinates['lat']}, lon={coordinates['lon']}")
            self.logger.info(f"📍 BBox: {coordinates['bbox']}")
            self.logger.info(f"🏷️ Display name: {coordinates.get('display_name', 'N/A')}")
        else:
            self.logger.warning(f"⚠️ Location NOT found after {len(tool_calls)} attempts")
            self.logger.warning(f"Tool calls made: {[tc['name'] for tc in tool_calls]}")
    
    async def search(self, query: str, reason: str = "", context: str = "") -> SearchResult:
        """
        Поиск координат через ReAct агента.
        
        Args:
            query: Запрос на поиск (например, "ЖК Скандинавия, бульвар Веласкеса, Коммунарка, Москва")
            reason: Зачем нужны координаты (для логирования и контекста)
            context: Дополнительный контекст об объекте
        
        Returns:
            SearchResult с результатами поиска
        """
        self.logger.info(f"🗺️ Location Agent: Starting search for '{query}'")
        if reason:
            self.logger.info(f"   💡 Reason: {reason}")
        if context:
            self.logger.info(f"   📝 Context: {context}")
        
        # Формируем полный запрос с системным промптом и дополнительным контекстом
        additional_context = ""
        if reason or context:
            additional_context = "\n\nДополнительный контекст:"
            if reason:
                additional_context += f"\n- Цель поиска: {reason}"
            if context:
                additional_context += f"\n- Информация об объекте: {context}"
        
        full_query = f"""
Найди координаты для локации: "{query}"{additional_context}
"""
        
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
            
            # Парсим координаты из ответа
            coordinates = self._parse_coordinates_from_response(response)
            
            # Логируем результат
            self._log_search_result(coordinates, tool_calls)
            
            if coordinates and coordinates.get("found"):
                return SearchResult(
                    found=True,
                    coordinates=coordinates,
                    messages=messages,
                    total_attempts=len(tool_calls),
                    tool_calls=tool_calls,
                    agent_thoughts=agent_thoughts,
                    error=None
                )
            else:
                return SearchResult(
                    found=False,
                    coordinates=None,
                    messages=messages,
                    total_attempts=len(tool_calls),
                    tool_calls=tool_calls,
                    agent_thoughts=agent_thoughts,
                    error="No coordinates found after all attempts"
                )
        
        except Exception as e:
            self.logger.error(f"❌ Error in Location Agent: {e}", exc_info=True)
            return SearchResult(
                found=False,
                coordinates=None,
                messages=[],
                total_attempts=0,
                tool_calls=[],
                agent_thoughts=[],
                error=str(e)
            )
