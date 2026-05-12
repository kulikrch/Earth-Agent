"""
Shared utilities for multi-agent system.

This module contains common functions used across multiple agents
(LocationAgent, DataAcquisitionAgent, etc.) to avoid code duplication.
"""

import json
import re
from typing import Any, Dict, List, Optional, Sequence


def extract_text_from_content(content: Sequence[Any]) -> str:
    """
    Извлечь текст из content (формат use_responses_api=True).
    
    Args:
        content: Последовательность элементов формата [{'type': 'text', 'text': '...'}]
                Использует Sequence для совместимости с list[str | dict[Unknown, Unknown]]
    
    Returns:
        Объединённый текст из всех текстовых блоков
        
    Note:
        Ожидается формат Responses API: content = [{'type': 'text', 'text': '...'}]
        Использует Sequence вместо List для ковариантности типов
        
    Example:
        >>> content = [{'type': 'text', 'text': 'Hello'}, {'type': 'text', 'text': ' World'}]
        >>> extract_text_from_content(content)
        'Hello World'
    """
    text_parts: List[str] = []
    for item in content:
        if isinstance(item, dict) and item.get('type') == 'text':
            text_parts.append(item.get('text', ''))
    return ''.join(text_parts)


def try_parse_json_with_required_key(text: str, required_key: str) -> Optional[Dict[str, Any]]:
    """
    Пытается извлечь JSON из текста используя несколько стратегий с проверкой обязательного ключа.
    
    Args:
        text: Текст для парсинга
        required_key: Обязательный ключ, который должен присутствовать в JSON
        
    Returns:
        Распарсенный JSON или None
        
    Example:
        >>> text = '```json\\n{"found": true, "lat": 55.5}\\n```'
        >>> try_parse_json_with_required_key(text, "found")
        {'found': True, 'lat': 55.5}
    """
    # Стратегия 1: JSON внутри markdown code block ```json ... ```
    markdown_match = re.search(r'```json\s*(\{.*?\})\s*```', text, re.DOTALL)
    if markdown_match:
        try:
            result = json.loads(markdown_match.group(1))
            if required_key in result:
                return result
        except json.JSONDecodeError:
            pass
    
    # Стратегия 2: Чистый JSON объект с обязательным ключом
    # Используем более умный regex который учитывает вложенность
    json_pattern = rf'\{{(?:[^{{}}]|(?:\{{[^{{}}]*\}}))*"{required_key}"(?:[^{{}}]|(?:\{{[^{{}}]*\}}))*\}}'
    json_match = re.search(json_pattern, text, re.DOTALL)
    if json_match:
        try:
            result = json.loads(json_match.group())
            if required_key in result:
                return result
        except json.JSONDecodeError:
            pass
    
    return None
