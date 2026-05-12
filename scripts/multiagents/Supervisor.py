import asyncio
import json
import logging
import re
from typing import Dict, List, Optional, Annotated, Any, Literal
from typing_extensions import TypedDict
from langchain_core.messages import HumanMessage, SystemMessage, BaseMessage
from langgraph.graph import StateGraph, END
import operator
from langchain_openai import ChatOpenAI
from .utils import extract_text_from_content


# ============================================================================
# MULTI-AGENT ARCHITECTURE: Supervisor
# ============================================================================

class SupervisorState(TypedDict):
    """Общее состояние для мультиагентной системы"""
    question: str
    question_type: str
    question_id: Optional[str]
    
    # Location Agent
    location_search_needed: bool
    location_query: Optional[str]
    location_result: Optional[Dict[str, Any]]
    
    # Data Acquisition Agent
    data_acquisition_needed: bool
    data_requirements: Optional[Dict[str, Any]]
    data_acquisition_result: Optional[Dict[str, Any]]
    analysis_contract: Optional[Dict[str, Any]]
    
    # Main Agent
    messages: Annotated[List[Dict[str, Any]], operator.add]
    next_agent: Optional[str]
    final_answer: Optional[str]
    main_agent_messages: Optional[List[BaseMessage]]  # Полные сообщения main agent для логирования
    answer_decision_check: Optional[str]
    
    # Metadata
    metadata: Dict[str, Any]


class MultiAgentSupervisor:
    """Супервизор для координации Location Agent, Data Acquisition Agent и Main Agent"""
    
    def __init__(
        self,
        llm: ChatOpenAI,
        analyze_question_agent: Any,
        location_agent: Any,
        data_acquisition_agent: Any,
        main_agent_executor: Any,
        system_prompt: str
    ) -> None:
        """
        Инициализация Supervisor.
        
        Args:
            llm: ChatOpenAI модель для анализа вопросов
            analyze_question_agent: Агент для анализа вопросов
            location_agent: Location Agent для геокодирования
            data_acquisition_agent: Data Acquisition Agent для скачивания данных
            main_agent_executor: Main Agent для обработки вопросов
            system_prompt: Системный промпт для Main Agent
        """
        self.llm: ChatOpenAI = llm
        self.analyze_question_agent: Any = analyze_question_agent
        self.location_agent: Any = location_agent
        self.data_acquisition_agent: Any = data_acquisition_agent
        self.main_agent_executor: Any = main_agent_executor
        self.system_prompt: str = system_prompt
        self.logger: logging.Logger = logging.getLogger("Supervisor")
        self.graph = self._create_supervisor_graph()
    
    
    def _create_supervisor_graph(self) -> Any:
        workflow = StateGraph(SupervisorState)
        workflow.add_node("analyze_question", self._analyze_question)
        workflow.add_node("call_location_agent", self._call_location_agent)
        workflow.add_node("call_data_acquisition_agent", self._call_data_acquisition_agent)
        workflow.add_node("call_main_agent", self._call_main_agent)
        workflow.add_node("finalize", self._finalize)
        
        workflow.set_entry_point("analyze_question")
        workflow.add_conditional_edges(
            "analyze_question",
            self._route_after_analysis,
            {
                "location_agent": "call_location_agent",
                "data_acquisition_agent": "call_data_acquisition_agent",
                "main_agent": "call_main_agent"
            }
        )
        workflow.add_edge("call_location_agent", "call_data_acquisition_agent")
        workflow.add_conditional_edges(
            "call_data_acquisition_agent",
            self._route_after_data_acquisition,
            {"main_agent": "call_main_agent", "error": "finalize"}
        )
        workflow.add_edge("call_main_agent", "finalize")
        workflow.add_edge("finalize", END)
        return workflow.compile()
    
    def _analyze_question(self, state: SupervisorState) -> SupervisorState:
        """
        Вызов Analyze Question Agent для анализа вопроса.
        
        Args:
            state: Текущее состояние Supervisor
            
        Returns:
            Обновлённое состояние с результатами анализа
        """
        question = state["question"]
        
        self.logger.info(f"🔍 Calling Analyze Question Agent")
        
        # Call Analyze Question Agent
        analysis = self.analyze_question_agent.analyze(question)
        
        self.logger.info(f"📊 Analyze Question Agent completed:")
        self.logger.info(f"   - Location needed: {analysis.get('location_needed', False)}")
        self.logger.info(f"   - Data acquisition needed: {analysis.get('data_acquisition_needed', False)}")
        
        # Update state with analysis results
        state["location_search_needed"] = analysis.get("location_needed", False)
        state["location_query"] = analysis.get("location_query")
        
        state["data_acquisition_needed"] = analysis.get("data_acquisition_needed", False)
        state["data_requirements"] = analysis.get("data_requirements")
        state["analysis_contract"] = analysis.get("analysis_contract")

        if state.get("analysis_contract"):
            self.logger.info(f"   - Analysis contract: {state['analysis_contract']}")
        
        # Save full analysis
        state["messages"].append({
            "agent": "analyze_question_agent",
            "action": "analysis",
            "content": analysis
        })
        
        return state
    
    def _route_after_analysis(self, state: SupervisorState) -> Literal["location_agent", "data_acquisition_agent", "main_agent"]:
        """Route after analyzing question"""
        # Priority: location_agent > data_acquisition_agent > main_agent
        if state.get("location_search_needed", False):
            return "location_agent"
        elif state.get("data_acquisition_needed", False):
            return "data_acquisition_agent"
        else:
            return "main_agent"
    
    def _route_after_data_acquisition(self, state: SupervisorState) -> Literal["main_agent", "error"]:
        """Route after data acquisition attempt"""
        # Always proceed to main_agent (even if acquisition failed)
        # Main agent can handle the case when files are not available
        return "main_agent"
    
    def _call_location_agent(self, state: SupervisorState) -> SupervisorState:
        """
        Вызов Location Agent для геокодирования.
        
        Args:
            state: Текущее состояние Supervisor
            
        Returns:
            Обновлённое состояние с результатами геокодирования
        """
        
        location_query = state.get("location_query") or self._extract_location_from_question(state["question"])
        
        # Извлекаем reason и context из analysis
        analysis = next((m["content"] for m in state["messages"] if m.get("action") == "analysis"), {})
        reason = analysis.get("reason", "Для поиска спутниковых данных")
        context = analysis.get("context", "")
        
        self.logger.info(f"🗺️ Calling Location Agent with query: '{location_query}'")
        if reason:
            self.logger.info(f"   Reason: {reason}")
        if context:
            self.logger.info(f"   Context: {context}")
        
        # Run async search in sync context
        location_result = asyncio.run(self.location_agent.search(
            query=location_query,
            reason=reason,
            context=context
        ))
        
        # Детальное логирование результатов Location Agent
        self.logger.info(f"📊 Location Agent completed:")
        self.logger.info(f"   - Found: {location_result.get('found', False)}")
        self.logger.info(f"   - Attempts: {location_result.get('total_attempts', 0)}")
        
        if location_result.get('found'):
            coords = location_result.get('coordinates', {})
            self.logger.info(f"   - Coordinates: lat={coords.get('lat')}, lon={coords.get('lon')}")
            self.logger.info(f"   - Display name: {coords.get('display_name', 'N/A')}")
        else:
            self.logger.warning(f"   - Error: {location_result.get('error', 'Unknown')}")
        
        # Логируем tool calls
        if 'tool_calls' in location_result:
            self.logger.info(f"   - Tool calls made:")
            for i, tc in enumerate(location_result['tool_calls'], 1):
                self.logger.info(f"     {i}. {tc['name']}({tc['args'][:100]}...)")
        
        # Логируем мысли агента
        if 'agent_thoughts' in location_result:
            self.logger.info(f"   - Agent thoughts:")
            for i, thought in enumerate(location_result['agent_thoughts'], 1):
                self.logger.info(f"     {i}. {thought[:150]}...")
        
        state["location_result"] = location_result
        state["messages"].append({
            "agent": "location_agent",
            "action": "search",
            "query": location_query,
            "result": location_result
        })
        return state
    
    def _extract_location_from_question(self, question: str) -> str:
        """
        Извлечь название локации из вопроса (fallback метод).
        
        Args:
            question: Текст вопроса
            
        Returns:
            Извлечённая строка локации
        """
        words = question.split()
        capitalized = [w for w in words if w and w[0].isupper() and len(w) > 3]
        return " ".join(capitalized[:3]) if capitalized else question
    
    def _call_data_acquisition_agent(self, state: SupervisorState) -> SupervisorState:
        """
        Вызов Data Acquisition Agent для скачивания спутниковых данных.
        
        Args:
            state: Текущее состояние Supervisor
            
        Returns:
            Обновлённое состояние с результатами скачивания данных
        """
        
        # Check if data acquisition is needed
        if not state.get("data_acquisition_needed", False):
            self.logger.info("📂 Data acquisition not needed, skipping Data Acquisition Agent")
            return state
        
        # Get requirements from analysis
        analysis = next((m["content"] for m in state["messages"] if m.get("action") == "analysis"), {})
        requirements_raw = analysis.get("data_requirements") if isinstance(analysis, dict) else {}
        requirements = dict(requirements_raw) if isinstance(requirements_raw, dict) else {}
        analysis_contract = state.get("analysis_contract")
        if isinstance(analysis_contract, dict):
            requirements["analysis_contract"] = analysis_contract

        # Use per-question output directory to avoid cross-question file mixing.
        question_id = str(state.get("question_id") or "").strip()
        if question_id:
            safe_question_id = re.sub(r"[^0-9A-Za-z_-]", "_", question_id)
            requirements["output_dir"] = f"question_{safe_question_id}"
        
        # Get bbox from Location Agent if available
        bbox = None
        location_result = state.get("location_result")
        if location_result and location_result.get("found"):
            coords = location_result["coordinates"]
            bbox = coords["bbox"]
            requirements["bbox"] = bbox
        
        # Format requirements as natural language
        requirements_text = self._format_data_requirements(requirements)
        
        self.logger.info("📡 Calling Data Acquisition Agent...")
        self.logger.info(f"   Requirements:\n{requirements_text}")
        
        # Run Data Acquisition Agent
        result = asyncio.run(self.data_acquisition_agent.search(requirements_text))
        
        self.logger.info(f"📊 Data Acquisition Agent completed:")
        self.logger.info(f"   - Success: {result.get('success', False)}")
        self.logger.info(f"   - Files downloaded: {len(result.get('downloaded_files', []))}")
        self.logger.info(f"   - Total attempts: {result.get('total_attempts', 0)}")
        
        if result.get('success'):
            for i, file_info in enumerate(result.get('downloaded_files', []), 1):
                self.logger.info(f"     {i}. {file_info.get('path', 'N/A')}")
        else:
            self.logger.warning(f"   - Error: {result.get('error', 'Unknown')}")
        
        state["data_acquisition_result"] = result
        state["messages"].append({
            "agent": "data_acquisition_agent",
            "action": "search",
            "result": result
        })
        
        return state
    
    def _format_data_requirements(self, requirements: Dict[str, Any]) -> str:
        """Format data requirements as natural language for Data Acquisition Agent"""
        text = "Найди и скачай спутниковые снимки для анализа:\n\n"
        
        # Location
        if "bbox" in requirements:
            text += f"Локация: bbox={requirements['bbox']}\n\n"
        
        # Time periods
        if "dates" in requirements and requirements["dates"]:
            text += "Временные периоды:\n"
            for i, date_period in enumerate(requirements["dates"], 1):
                label = date_period.get("label", f"Period {i}")
                start = date_period.get("start", "")
                end = date_period.get("end", "")
                text += f"  {i}. {label}: {start} - {end}\n"
            text += "\n"

        # Purpose
        if "purpose" in requirements:
            text += f"Цель анализа: {requirements['purpose']}\n\n"

        analysis_contract = requirements.get("analysis_contract")
        if isinstance(analysis_contract, dict) and analysis_contract:
            text += "Аналитический контракт для выбора данных:\n"
            text += json.dumps(analysis_contract, ensure_ascii=False, indent=2)
            text += "\n\n"
        
        # Output directory
        output_dir = requirements.get("output_dir", "output")
        text += f"Сохранить файлы в директорию: {output_dir}/\n"
        
        return text
    
    def _call_main_agent(self, state: SupervisorState) -> SupervisorState:
        """
        Вызов Main Agent для обработки вопроса.
        
        Args:
            state: Текущее состояние Supervisor
            
        Returns:
            Обновлённое состояние с финальным ответом
        """
        question = state["question"]
        
        # Build context from previous agents
        context_parts = []
        
        # Location context
        location_result = state.get("location_result")
        if location_result and location_result.get("found"):
            coords = location_result["coordinates"]
            context_parts.append(f"""
ИНФОРМАЦИЯ О ЛОКАЦИИ (получена от Location Agent):
- Адрес: {coords.get('display_name', 'N/A')}
- Координаты: lat={coords['lat']}, lon={coords['lon']}
- Bounding Box: {coords['bbox']}""")

        analysis_contract = state.get("analysis_contract")
        if isinstance(analysis_contract, dict) and analysis_contract:
            context_parts.append(
                "\nАНАЛИТИЧЕСКИЙ КОНТРАКТ (получен от AnalyzeQuestionAgent):\n"
                + json.dumps(analysis_contract, ensure_ascii=False, indent=2)
                + "\n\n"
                + "Следуй этому контракту при выборе инструментов анализа. "
                + "Не заменяй измеряемую величину другой величиной: если expected_unit=percent_of_area, "
                + "нужно считать долю площади/пикселей, а не среднее значение raw-канала."
            )
        
        # Data acquisition context
        data_acquisition_result = state.get("data_acquisition_result")
        if data_acquisition_result:
            result = data_acquisition_result
            if result.get("success"):
                files = result.get("downloaded_files", [])
                channel_summary = result.get("channel_summary", [])
                
                # Формируем детальную информацию о скачанных файлах и каналах
                files_info = []
                for f in files:
                    file_details = [f"  - path: {f.get('path', 'N/A')}"]
                    for key in ("band", "date", "period_label", "semantic_role", "analysis_use"):
                        value = f.get(key)
                        if value:
                            file_details.append(f"    {key}: {value}")
                    file_line = "\n".join(file_details)
                    files_info.append(file_line)
                
                # Формируем объяснение по каналам
                channel_explanations = []
                if channel_summary:
                    channel_explanations.append("\n📡 ИНФОРМАЦИЯ О СПЕКТРАЛЬНЫХ КАНАЛАХ:")
                    for ch_info in channel_summary:
                        channel_explanations.append(f"\n{ch_info.get('band', 'No band')}\n{ch_info.get('reasoning', 'No reasoning')}")
                
                context_parts.append(f"""
ДАННЫЕ СКАЧАНЫ (Data Acquisition Agent):
- Количество файлов: {len(files)}
- Коллекция: {result.get("collection", "N/A")}
- Файлы доступны для анализа:
""" + "\n".join(files_info) + "\n".join(channel_explanations))
            else:
                context_parts.append(f"""
ПРЕДУПРЕЖДЕНИЕ: Data Acquisition Agent не смог скачать данные.
Причина: {result.get('error', 'Unknown')}
Попробуй использовать локальные данные через get_filelist.""")
        
        # Combine query
        if context_parts:
            full_context = "\n".join(context_parts)
            full_query = f"Question: {question}\n{full_context}"
        else:
            full_query = f"Question: {question}"
        
        # Create HumanMessage for main agent
        messages: List[BaseMessage] = [
            SystemMessage(content=self.system_prompt),
            HumanMessage(content=full_query)
        ]
        
        try:
            self.logger.info("🤖 Calling Main Agent")

            async def _run_main_agent():
                return await self.main_agent_executor.ainvoke(
                    {"messages": messages},
                    {"recursion_limit": 40}
                )

            # Run async main agent in sync context
            result = asyncio.run(_run_main_agent())
            final_answer = self._extract_answer_from_result(
                result,
                multiple_choice=state.get("question_type") == "multiple_choice",
            )
            if (
                state.get("question_type") == "multiple_choice"
                and not self._is_strict_multiple_choice_answer(final_answer)
            ):
                self.logger.warning("⚠️ Main Agent returned malformed answer, requesting one format-only repair")
                repair_result = asyncio.run(self._repair_malformed_answer(result))
                if repair_result is not None:
                    result = repair_result
                    final_answer = self._extract_answer_from_result(result, multiple_choice=True)
            decision_check = self._extract_decision_check_from_result(result)
            state["final_answer"] = final_answer
            state["answer_decision_check"] = decision_check
            
            # Сохраняем ПОЛНЫЕ сообщения main agent для логирования
            state["main_agent_messages"] = result.get("messages", [])
            
            # В supervisor messages добавляем только краткую запись
            state["messages"].append({
                "agent": "main_agent", 
                "action": "analyze", 
                "answer": final_answer,
                "decision_check": decision_check,
                "query_preview": full_query[:200]  # Just first 200 chars for context
            })
        except Exception as e:
            import traceback
            error_details = traceback.format_exc()
            print(f"Error in main agent: {error_details}")
            state["final_answer"] = f"<Answer>Ошибка: {str(e)}</Answer>"
            state["answer_decision_check"] = None
            state["main_agent_messages"] = []
            state["messages"].append({"agent": "main_agent", "action": "error", "error": str(e)})
        
        return state
    
    @staticmethod
    def _normalize_multiple_choice_answer(text: Any) -> str:
        """Normalize model output to strict <Answer>A</Answer> format when possible."""
        if text is None:
            return "FAIL"

        raw = str(text).strip()
        if not raw:
            return "UNKNOWN"

        letter_map = {
            "А": "A",
            "В": "B",
            "С": "C",
            "Д": "D",
            "Е": "E",
            "Ф": "F",
        }
        normalized = raw
        for cyr, lat in letter_map.items():
            normalized = normalized.replace(cyr, lat)

        answer_match = re.search(r"<Answer>(.*?)</Answer>", normalized, re.IGNORECASE | re.DOTALL)
        answer_text = answer_match.group(1).strip() if answer_match else normalized

        patterns = (
            r"^\s*([A-F])\s*$",
            r"\(\s*([A-F])\s*\)",
            r"(?:вариант|option|answer|ответ|выбор)\s*[:\-]?\s*([A-F])\b",
            r"^\s*([A-F])\s*[\.\):\-]",
            r"\b([A-F])\b",
        )
        for pattern in patterns:
            match = re.search(pattern, answer_text, re.IGNORECASE)
            if match:
                return f"<Answer>{match.group(1).upper()}</Answer>"

        return raw

    @staticmethod
    def _extract_last_ai_text(result: Any) -> str:
        if not isinstance(result, dict):
            return str(result)

        messages = result.get("messages", [])
        for message in reversed(messages):
            if hasattr(message, "type") and message.type == "ai":
                content = getattr(message, "content", "")
                if isinstance(content, list):
                    return extract_text_from_content(content)
                return str(content) if content else ""
        return str(result)

    @staticmethod
    def _is_strict_multiple_choice_answer(answer: Any) -> bool:
        return bool(re.fullmatch(r"\s*<Answer>[A-F]</Answer>\s*", str(answer or ""), re.IGNORECASE))

    async def _repair_malformed_answer(self, result: Any) -> Optional[Any]:
        """Ask the model once to reformat a malformed multiple-choice answer."""
        if not isinstance(result, dict):
            return None

        messages = list(result.get("messages", []))
        if not messages:
            return None

        repair_prompt = (
            "Твой предыдущий финальный ответ имеет неверный формат. "
            "Не вызывай инструменты и не пересчитывай ничего. "
            "Верни только одну латинскую букву выбранного варианта в формате "
            "<Answer>A</Answer>, <Answer>B</Answer>, <Answer>C</Answer> или <Answer>D</Answer>."
        )
        messages.append(HumanMessage(content=repair_prompt))

        try:
            return await self.main_agent_executor.ainvoke(
                {"messages": messages},
                {"recursion_limit": 5},
            )
        except Exception as exc:
            self.logger.warning(f"⚠️ Answer format repair failed: {exc}")
            return None

    def _extract_decision_check_from_result(self, result: Any) -> Optional[str]:
        """Extract DECISION_CHECK block from the final Main Agent response if present."""
        content_text = self._extract_last_ai_text(result)
        if not content_text:
            return None

        match = re.search(
            r"DECISION_CHECK\s*:\s*(.*?)(?:<Answer>|$)",
            content_text,
            flags=re.IGNORECASE | re.DOTALL,
        )
        if not match:
            return None

        decision_check = match.group(1).strip()
        return decision_check or None

    def _extract_answer_from_result(self, result: Any, multiple_choice: bool = True) -> str:
        """
        Извлечь финальный ответ из результата Main Agent.
        
        Args:
            result: Результат выполнения Main Agent
            
        Returns:
            Извлечённый ответ в формате <Answer>...</Answer>
        """
        content_text = self._extract_last_ai_text(result)
        answer_match = re.search(r'<Answer>(.*?)</Answer>', content_text, re.IGNORECASE | re.DOTALL)
        if answer_match:
            if not multiple_choice:
                return f"<Answer>{answer_match.group(1).strip()}</Answer>"
            return self._normalize_multiple_choice_answer(answer_match.group(1))
        if not multiple_choice:
            return content_text
        return self._normalize_multiple_choice_answer(content_text)
    
    def _finalize(self, state: SupervisorState) -> SupervisorState:
        """
        Финализация выполнения и сбор метаданных.
        
        Args:
            state: Текущее состояние Supervisor
            
        Returns:
            Финальное состояние с метаданными
        """
        location_result = state.get("location_result", {})
        data_acquisition_result = state.get("data_acquisition_result", {})
        state["metadata"] = {
            "location_search_used": state.get("location_search_needed", False),
            "location_found": location_result.get("found", False) if location_result else False,
            "data_acquisition_used": state.get("data_acquisition_needed", False),
            "data_acquisition_successful": data_acquisition_result.get("success", False) if data_acquisition_result else False,
            "analysis_contract": state.get("analysis_contract"),
            "answer_decision_check": state.get("answer_decision_check"),
            "total_steps": len(state["messages"])
        }
        state["messages"].append({"agent": "supervisor", "action": "finalize", "metadata": state["metadata"]})
        return state
    
    async def run(
        self,
        question: str,
        question_type: str = "open_ended",
        question_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Запустить мультиагентную систему для обработки вопроса.
        
        Args:
            question: Вопрос пользователя
            question_type: Тип вопроса ("open_ended" или "multiple_choice")
            
        Returns:
            Словарь с ответом, метаданными и сообщениями агентов
        """
        initial_state = SupervisorState(
            question=question,
            question_type=question_type,
            question_id=question_id,
            location_search_needed=False,
            location_query=None,
            location_result=None,
            data_acquisition_needed=False,
            data_requirements=None,
            data_acquisition_result=None,
            analysis_contract=None,
            messages=[],
            next_agent=None,
            final_answer=None,
            answer_decision_check=None,
            metadata={},
            main_agent_messages=None
        )
        final_state = await asyncio.to_thread(self.graph.invoke, initial_state)
        return {
            "answer": final_state.get("final_answer"),
            "metadata": final_state.get("metadata", {}),
            "messages": final_state.get("messages", []),
            "location_result": final_state.get("location_result"),
            "analysis_contract": final_state.get("analysis_contract"),
            "answer_decision_check": final_state.get("answer_decision_check"),
            "main_agent_messages": final_state.get("main_agent_messages", [])
        }
