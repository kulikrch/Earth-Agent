"""
Analyze Question Agent - Specialized agent for question analysis

This agent is responsible for analyzing Earth science questions to determine:
1. Whether location search is needed (geographic entities mentioned)
2. Whether data acquisition is needed (no local data path provided)
3. Extracting location queries with full context
4. Formulating data requirements for satellite imagery

Architecture:
- LLM-based analysis (no ReAct, just single LLM call)
- Input: Question text
- Output: Structured analysis (location_needed, data_acquisition_needed, requirements)
"""

import json
import re
from typing import Dict, Any
from langchain_core.messages import SystemMessage, HumanMessage as CoreHumanMessage
from langchain_openai import ChatOpenAI


class AnalyzeQuestionAgent:
    """
    Специализированный агент для анализа вопросов о дистанционном зондировании Земли.
    
    Анализирует вопрос и определяет:
    - Нужен ли поиск координат (location_needed)
    - Нужно ли скачивание данных (data_acquisition_needed)
    - Формирует запросы и требования для других агентов
    """
    
    def __init__(self, llm: ChatOpenAI):
        """
        Initialize Analyze Question Agent.
        
        Args:
            llm: ChatOpenAI model (gpt-4o-mini recommended for cost efficiency)
        """
        self.llm = llm
        
        # Load system prompt from file (like Location Agent)
        with open('scripts/prompts/analyze_question_agent.md', 'r', encoding='utf-8') as f:
            self.system_prompt = f.read()
    
    @staticmethod
    def _extract_text_from_response(response) -> str:
        """Извлечь текст из ответа LLM (совместимо с use_responses_api=True)
        
        При use_responses_api=True формат: response.content = [{'type': 'text', 'text': '...'}]
        При обычном режиме: response.content = "..."
        """
        content = response.content
        
        # Если content - список словарей (Responses API)
        if isinstance(content, list):
            text_parts = []
            for item in content:
                if isinstance(item, dict) and item.get('type') == 'text':
                    text_parts.append(item.get('text', ''))
            return ''.join(text_parts)
        
        # Если content - строка (обычный режим)
        return str(content)
    
    def analyze(self, question: str) -> Dict[str, Any]:
        """
        Analyze question to determine routing and requirements.
        
        Args:
            question: The question text to analyze
        
        Returns:
            Dict with analysis results:
                {
                    "location_needed": bool,
                    "location_query": str | None,
                    "reason": str | None,
                    "context": str | None,
                    "data_acquisition_needed": bool,
                    "data_requirements": dict | None,
                    "analysis_contract": dict | None
                }
        """
        import logging
        logger = logging.getLogger("AnalyzeQuestionAgent")
        
        logger.info(f"🔍 Analyze Question Agent: Starting analysis")
        logger.info(f"   Question preview: {question[:200]}...")
        
        try:
            # Invoke LLM with system prompt
            messages = [
                SystemMessage(content=self.system_prompt),
                CoreHumanMessage(content=f"Вопрос: {question}")
            ]
            
            logger.info(f"🤖 Invoking LLM for analysis")
            response = self.llm.invoke(messages)
            
            # Extract text from response
            response_text = self._extract_text_from_response(response)
            
            logger.info(f"💭 LLM response preview: {response_text[:300]}...")
            
            # Parse JSON from response
            json_match = re.search(r'\{.*\}', response_text, re.DOTALL)
            if not json_match:
                logger.error("❌ No JSON found in LLM response")
                raise ValueError("No JSON found in analysis response")
            
            analysis = json.loads(json_match.group())
            analysis = self._normalize_analysis(analysis)
            
            # Log analysis results
            logger.info(f"✅ Analysis completed:")
            logger.info(f"   - Location needed: {analysis.get('location_needed', False)}")
            if analysis.get('location_needed'):
                logger.info(f"   - Location query: {analysis.get('location_query', 'N/A')}")
                logger.info(f"   - Reason: {analysis.get('reason', 'N/A')}")
            
            logger.info(f"   - Data acquisition needed: {analysis.get('data_acquisition_needed', False)}")
            if analysis.get('data_acquisition_needed'):
                requirements = analysis.get('data_requirements', {})
                logger.info(f"   - Data requirements:")
                logger.info(f"      - Purpose: {requirements.get('purpose', 'N/A')}")
                logger.info(f"      - Output dir: {requirements.get('output_dir', 'N/A')}")
                dates = requirements.get('dates')
                if isinstance(dates, list):
                    logger.info(f"      - Time periods: {len(dates)}")
                elif dates is None:
                    logger.info("      - Time periods: N/A")
            
            return analysis
        
        except json.JSONDecodeError as e:
            logger.error(f"❌ Error parsing JSON from LLM response: {e}")
            logger.error(f"Response was: {response_text}")
            
            # Return fallback defaults
            return {
                "location_needed": False,
                "location_query": None,
                "reason": None,
                "context": None,
                "data_acquisition_needed": False,
                "data_requirements": None,
                "analysis_contract": None,
                "error": f"JSON parsing error: {str(e)}"
            }
        
        except Exception as e:
            logger.error(f"❌ Error in Analyze Question Agent: {e}", exc_info=True)
            
            # Return fallback defaults
            return {
                "location_needed": False,
                "location_query": None,
                "reason": None,
                "context": None,
                "data_acquisition_needed": False,
                "data_requirements": None,
                "analysis_contract": None,
                "error": str(e)
            }

    @staticmethod
    def _normalize_analysis(analysis: Dict[str, Any]) -> Dict[str, Any]:
        """
        Normalize LLM JSON response to a safe schema without domain-specific hardcoded rules.
        """
        normalized: Dict[str, Any] = {
            "location_needed": bool(analysis.get("location_needed", False)),
            "location_query": analysis.get("location_query"),
            "reason": analysis.get("reason"),
            "context": analysis.get("context"),
            "data_acquisition_needed": bool(analysis.get("data_acquisition_needed", False)),
            "data_requirements": analysis.get("data_requirements"),
            "analysis_contract": analysis.get("analysis_contract"),
        }

        # Normalize optional string fields
        for field in ("location_query", "reason", "context"):
            value = normalized.get(field)
            normalized[field] = value.strip() if isinstance(value, str) and value.strip() else None

        requirements = normalized.get("data_requirements")
        if requirements is not None and not isinstance(requirements, dict):
            requirements = None

        if isinstance(requirements, dict):
            req = dict(requirements)
            dates = req.get("dates")
            if dates is not None and not isinstance(dates, list):
                req["dates"] = None
            normalized["data_requirements"] = req
        else:
            normalized["data_requirements"] = None

        contract = normalized.get("analysis_contract")
        if isinstance(contract, dict):
            normalized["analysis_contract"] = dict(contract)
        else:
            normalized["analysis_contract"] = None

        return normalized
