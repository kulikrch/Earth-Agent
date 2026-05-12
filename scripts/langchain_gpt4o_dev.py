import os
import sys

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from multiagents.location_agent import LocationAgent
from multiagents.Supervisor import MultiAgentSupervisor
from multiagents.data_acquisition_agent import DataAcquisitionAgent
from multiagents.AnalyzeQuestionAgent import AnalyzeQuestionAgent
os.environ["GTIFF_SRS_SOURCE"]="EPSG"

# Load environment variables from .env file
from dotenv import load_dotenv
load_dotenv()

import json
import logging
import asyncio
import argparse
import re
from enum import auto
from tqdm import tqdm
from pathlib import Path
from copy import deepcopy
from datetime import datetime
from logging.handlers import RotatingFileHandler

from langchain_mcp_adapters.client import MultiServerMCPClient
from langgraph.prebuilt import create_react_agent
from langchain_openai import ChatOpenAI
from langchain.schema import HumanMessage

# Pprint for debugging
from pprint import pprint
# Keep working directory as repository root (don't change to scripts/)
# os.chdir(os.path.dirname(os.path.abspath(__file__)))
# This allows relative paths in config to work correctly

# Global variables
logger = None
temp_dir_path = None

# Configuration
model_name = 'gpt-4o'
autoplanning = False

# Development dataset mode: use development_quest.json and dev-tools only
USE_DEV_DATASET = True
DEV_TEST_QUESTION_IDS = []  # Empty means "run all development questions"

# Multi-agent mode: use Location Agent for geocoding
USE_MULTIAGENT = False  # Enable via --use-multiagent flag

# Системные промпты для разных типов вопросов
SYSTEM_PROMPT_MULTIPLE_CHOICE = '''
Вы — специалист по наукам о Земле, и вам необходимо использовать инструменты для ответа на вопросы с множественным выбором об анализе данных дистанционного зондирования Земли. Обратите внимание: если инструмент возвращает ошибку, вы можете попытаться снова только один раз. В конечном итоге вам нужно явно указать только правильный вариант ответа.

ВНИМАНИЕ:
1. Когда инструмент возвращает "Result saved at /path/to/file", вы должны использовать полный возвращенный путь "/path/to/file" во всех последующих вызовах инструментов.
2. ВАЖНО: Используйте get_filelist для проверки доступных файлов перед анализом. НЕ угадывайте имена файлов!
3. НИКОГДА не конструируйте путь вручную (например, "question3/file.tif"). Используйте только точные пути из tool-result.
4. Ваш финальный ответ должен быть в формате:
<Answer>A</Answer>
или <Answer>B</Answer>, <Answer>C</Answer>, <Answer>D</Answer>. Внутри тега должна быть только одна латинская буква варианта без пояснений.
5. В одном шаге вызывайте ТОЛЬКО ОДИН инструмент. Не делайте параллельные/зависимые tool-calls в одном ответе.

РАБОЧИЙ ПРОЦЕСС:
- Используйте get_filelist для проверки доступных файлов данных
- Обработайте файлы используя доступные инструменты анализа (Index, Analysis, Inversion, Perception, Statistics)
- Если файлы недоступны - сообщите об этом (предыдущие агенты должны были подготовить данные)

АНАЛИТИЧЕСКИЙ КОНТРАКТ:
- Если в вопросе есть блок "АНАЛИТИЧЕСКИЙ КОНТРАКТ", он задаёт измеряемую величину, единицы результата и тип сравнения.
- Не подменяйте измеряемую величину другой величиной.
- Если expected_unit=percent_of_area, нужно считать долю площади/пикселей, удовлетворяющих условию, а не среднее значение raw-канала или среднее значение индекса.
- Для нескольких годов/периодов с expected_unit=percent_of_area используйте инструмент, который возвращает долю отдельно для каждого raster (например calculate_batch_threshold_ratio). Не используйте calculate_threshold_ratio со списком файлов, если дальше нужно сравнить годы: он возвращает среднее по списку.
- Если expected_unit=celsius_difference, нужно считать температуру поверхности или разницу температур, а не долю площади или среднее значение отражательной способности.
- Для measurement_target=built_up_area_share используйте SWIR+NIR и сначала постройте built-up proxy/index (например NDBI), затем считайте долю пикселей/площади. Не используйте Red+NIR и не threshold'ьте raw-каналы как застроенную территорию.
- Для measurement_target=vegetation_area_share используйте NIR+Red и сначала постройте vegetation proxy/index (например NDVI), затем считайте долю пикселей/площади.
- Для measurement_target=water_area_share используйте water proxy/index из подходящих каналов, затем считайте долю пикселей/площади.
- Для measurement_target=surface_temperature_difference используйте LST/thermal-инструменты и температурные каналы; Sentinel-2 SWIR-каналы не являются thermal.
- Перед финальным ответом проверьте: (1) что посчитанная величина совпадает с measurement_target, (2) что единицы совпадают с expected_unit, (3) что выбранный вариант ответа соответствует именно этой величине.
- Если проверка показывает, что была посчитана не та величина, сделайте одну корректирующую попытку с подходящими инструментами.

ПРОТОКОЛ ВЫБОРА ОТВЕТА:
Перед финальным тегом <Answer> обязательно напишите блок DECISION_CHECK. Используйте гибкую структуру, не привязанную к двум периодам:
DECISION_CHECK:
- metric_name: название реально посчитанной величины
- expected_unit: единица из аналитического контракта
- computed_values: список или словарь всех чисел, нужных для выбора ответа. Ключи должны соответствовать вопросу: годы, периоды, локации, классы или сравниваемые объекты. Если вопрос не содержит нескольких периодов, не создавайте искусственные period_1/period_2.
- comparisons: разности, отношения, процентные пункты или относительные проценты, только если они нужны для выбора ответа
- option_A_claim: числовое/логическое утверждение варианта A
- option_B_claim: числовое/логическое утверждение варианта B
- option_C_claim: числовое/логическое утверждение варианта C
- option_D_claim: числовое/логическое утверждение варианта D
- closest_option: A/B/C/D
- decision_reason: почему выбранный вариант ближе к рассчитанным значениям

Правила выбора:
- Сравнивайте рассчитанные значения с каждым вариантом, а не выбирайте только по направлению изменения.
- Различайте процентные пункты и относительный процент: absolute_delta_pp = after_share - before_share; relative_change_percent = absolute_delta_pp / before_share * 100.
- Если варианты содержат приблизительные интервалы, выбирайте ближайший по рассчитанному числу вариант.
- Если рассчитанная величина не совпадает с вариантом/контрактом, сделайте одну корректирующую попытку до финального ответа.
- После DECISION_CHECK финальный ответ должен быть строго одной буквой в теге, например <Answer>C</Answer>.

ВАЖНО: Ваша задача - АНАЛИЗ данных. Поиск координат и скачивание данных выполняется другими агентами.
'''

SYSTEM_PROMPT_OPEN_ENDED = '''
Вы — специалист по наукам о Земле, и вам необходимо использовать инструменты для ответа на вопросы об анализе данных дистанционного зондирования Земли. Обратите внимание: если инструмент возвращает ошибку, вы можете попытаться снова только один раз.

ВНИМАНИЕ:
1. Когда инструмент возвращает "Result saved at /path/to/file", вы должны использовать полный возвращенный путь "/path/to/file" во всех последующих вызовах инструментов.
2. ВАЖНО: Используйте get_filelist для проверки доступных файлов перед анализом. НЕ угадывайте имена файлов!
3. НИКОГДА не конструируйте путь вручную (например, "question3/file.tif"). Используйте только точные пути из tool-result.
4. Предоставьте детальный, хорошо обоснованный ответ на основе вашего анализа. Включите конкретные значения, статистику и наблюдения из использования инструментов. Ваш финальный ответ должен быть обёрнут в теги:
<Answer>Ваш детальный ответ здесь</Answer>
5. В одном шаге вызывайте ТОЛЬКО ОДИН инструмент. Не делайте параллельные/зависимые tool-calls в одном ответе.

РАБОЧИЙ ПРОЦЕСС:
- Используйте get_filelist для проверки доступных файлов данных
- Обработайте файлы используя доступные инструменты анализа (Index, Analysis, Inversion, Perception, Statistics)
- Если файлы недоступны - сообщите об этом (предыдущие агенты должны были подготовить данные)

АНАЛИТИЧЕСКИЙ КОНТРАКТ:
- Если в вопросе есть блок "АНАЛИТИЧЕСКИЙ КОНТРАКТ", он задаёт измеряемую величину, единицы результата и тип сравнения.
- Не подменяйте измеряемую величину другой величиной.
- Если expected_unit=percent_of_area, нужно считать долю площади/пикселей, удовлетворяющих условию, а не среднее значение raw-канала или среднее значение индекса.
- Для нескольких годов/периодов с expected_unit=percent_of_area используйте инструмент, который возвращает долю отдельно для каждого raster (например calculate_batch_threshold_ratio). Не используйте calculate_threshold_ratio со списком файлов, если дальше нужно сравнить годы: он возвращает среднее по списку.
- Если expected_unit=celsius_difference, нужно считать температуру поверхности или разницу температур, а не долю площади или среднее значение отражательной способности.
- Для measurement_target=built_up_area_share используйте SWIR+NIR и сначала постройте built-up proxy/index (например NDBI), затем считайте долю пикселей/площади. Не используйте Red+NIR и не threshold'ьте raw-каналы как застроенную территорию.
- Для measurement_target=vegetation_area_share используйте NIR+Red и сначала постройте vegetation proxy/index (например NDVI), затем считайте долю пикселей/площади.
- Для measurement_target=water_area_share используйте water proxy/index из подходящих каналов, затем считайте долю пикселей/площади.
- Для measurement_target=surface_temperature_difference используйте LST/thermal-инструменты и температурные каналы; Sentinel-2 SWIR-каналы не являются thermal.
- Перед финальным ответом проверьте: (1) что посчитанная величина совпадает с measurement_target, (2) что единицы совпадают с expected_unit, (3) что ответ основан именно на этой величине.
- Если проверка показывает, что была посчитана не та величина, сделайте одну корректирующую попытку с подходящими инструментами.

ПРОТОКОЛ ОБОСНОВАНИЯ:
Перед финальным тегом <Answer> напишите блок DECISION_CHECK:
- metric_name: название реально посчитанной величины
- expected_unit: единица из аналитического контракта
- computed_values: список или словарь всех чисел, нужных для ответа. Ключи должны соответствовать вопросу: годы, периоды, локации, классы или сравниваемые объекты. Если вопрос не содержит нескольких периодов, не создавайте искусственные period_1/period_2.
- comparisons: разности, отношения, процентные пункты или относительные проценты, только если они нужны для ответа
- decision_reason: почему вывод следует из рассчитанных значений

ВАЖНО: Ваша задача - АНАЛИЗ данных. Поиск координат и скачивание данных выполняется другими агентами.
'''

def init_global_params():
    """Initialize global parameters and logging"""
    global temp_dir_path, logger
    
    if temp_dir_path is None:
        suffix = '_dev' if USE_DEV_DATASET else ''
        temp_dir_path = Path('./evaluate_langchain/{}{}_{}_{}'.format(
            model_name,
            suffix,
            'AP' if autoplanning else "IF", 
            datetime.now().strftime('%y-%m-%d_%H-%M')
        )).absolute()
    temp_dir_path.mkdir(parents=True, exist_ok=True)

    # Use simple file handler instead of complex JSON formatter
    logger = logging.getLogger("text_logger")
    logger.setLevel(logging.INFO)
    
    # Remove all existing handlers
    logger.handlers.clear()
    
    # Create file handler with UTF-8 encoding
    log_file = temp_dir_path / "{}_{}_langchain.json".format(
        model_name, 'AP' if autoplanning else "IF"
    )
    handler = logging.FileHandler(log_file, mode='w', encoding='utf-8')
    
    # Simple formatter - we'll write JSON manually in log calls
    handler.setFormatter(logging.Formatter('%(message)s'))
    logger.addHandler(handler)
    
    # Open JSON array
    with open(log_file, 'w', encoding='utf-8') as f:
        f.write('[\n')
    
    return temp_dir_path, logger


def init_chat_logger():
    """Initialize chat logger for .chat file like AgentScope"""
    global temp_dir_path
    chat_log_path = temp_dir_path / "{}_{}_langchain.chat".format(
        model_name, 'AP' if autoplanning else "IF"
    )
    return chat_log_path


def save_chat_message(chat_log_path, message_data):
    """Save a single chat message to .chat file in AgentScope format"""
    import time
    from datetime import datetime
    import uuid
    
    # Convert content to serializable format
    content = message_data.get('content', [])
    if not isinstance(content, (list, dict, str, int, float, bool, type(None))):
        # If content is not JSON-serializable, convert to string
        content = str(content)
    
    # Convert metadata to serializable format
    metadata = message_data.get('metadata', None)
    if metadata and not isinstance(metadata, (dict, str, int, float, bool, type(None))):
        metadata = str(metadata)
    
    # Format message in AgentScope style
    chat_record = {
        "__module__": "langchain.schema.messages",
        "__name__": "ChatMessage", 
        "id": str(uuid.uuid4()).replace('-', ''),
        "name": message_data.get('name', 'langchain_agent'),
        "role": message_data.get('role', 'assistant'),
        "content": content,
        "metadata": metadata,
        "timestamp": datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    }
    
    # Append to chat file (one JSON per line, like AgentScope)
    try:
        with open(chat_log_path, 'a', encoding='utf-8') as f:
            f.write(json.dumps(chat_record, ensure_ascii=False) + '\n')
    except TypeError as e:
        # If still not serializable, log error and skip
        print(f"Warning: Could not serialize chat message: {e}")
        print(f"Message data: {message_data}")


def load_langchain_config(config_path='agent/config_gpt4o.json'):
    """Load configuration and initialize LangChain components"""
    with open(config_path, 'r') as f:
        config = json.load(f)
    
    # Initialize OpenAI model with stricter parameters
    model_config = config['models'][0]
    api_key = model_config.get('api_key') or os.getenv('OPENAI_API_KEY')
    base_url = model_config.get('client_args', {}).get('base_url') or os.getenv('OPENAI_BASE_URL')
    llm_kwargs = {
        'model': model_config['model_name'],
        'api_key': api_key,
        'base_url': base_url,
        'temperature': 0.1,  # Lower temperature for more focused responses
        'request_timeout': 120,  # 2 minute timeout per request,
        'use_responses_api': True,
        'use_previous_response_id': True
    }
    
    # Add proxy support if configured
    # Note: langchain-openai automatically picks up OPENAI_PROXY/HTTP_PROXY/HTTPS_PROXY
    # from environment, so we just check and log if present
    proxy = os.getenv('OPENAI_PROXY') or os.getenv('HTTPS_PROXY') or os.getenv('HTTP_PROXY')
    if proxy:
        print(f"Using proxy: {proxy}")
        # Use the built-in openai_proxy parameter instead of custom http_client
        llm_kwargs['openai_proxy'] = proxy
    
    # Add generate_args via extra_body if present in config
    if 'generate_args' in model_config:
        llm_kwargs['extra_body'] = model_config['generate_args']
    
    llm = ChatOpenAI(**llm_kwargs)
    
    # Prepare MCP servers configuration
    mcp_servers = {}
    for server_name, server_config in config['mcpServers'].items():
        # Update paths to use current temp directory
        updated_args = []
        for arg in server_config['args']:
            if 'tmp/tmp/out' in arg:
                updated_args.append(str(temp_dir_path / 'out'))
            elif arg.startswith('tools/'):
                updated_args.append('agent/' + arg)
            else:
                updated_args.append(arg)
        
        # Use venv Python for Python MCP servers
        command = server_config['command']
        if command == 'python':
            # Use the same Python interpreter that's running this script
            command = sys.executable
        
        # Pass environment variables to subprocess
        server_env = dict(os.environ)  # Copy current environment
        # Force UTF-8 encoding and disable ANSI colors for cleaner logs in Tee-Object files.
        server_env['PYTHONIOENCODING'] = 'utf-8'
        server_env['NO_COLOR'] = '1'
        server_env['RICH_NO_COLOR'] = '1'
        server_env['PY_COLORS'] = '0'
        server_env['CLICOLOR'] = '0'
        
        mcp_servers[server_name] = {
            "command": command,
            "args": updated_args,
            "transport": "stdio",
            "env": server_env  # Pass env to subprocess
        }
    
    # Conditionally add EarthEngine MCP server if GEE credentials are set
    gee_key = os.getenv('GEE_SERVICE_ACCOUNT_KEY') or os.getenv('GEE_SERVICE_ACCOUNT_KEY_JSON')
    if gee_key:
        print(f"EarthEngine MCP enabled (key: {gee_key[:50]}...)")
        
        # Pass GEE key via command line arg (more reliable than env for subprocess)
        ee_args = ["agent/tools/EarthEngine.py", "--temp_dir", str(temp_dir_path / 'out')]
        if os.getenv('GEE_SERVICE_ACCOUNT_KEY'):
            ee_args.extend(["--gee_key", os.getenv('GEE_SERVICE_ACCOUNT_KEY')])
        
        ee_env = dict(os.environ)
        ee_env['PYTHONIOENCODING'] = 'utf-8'  # Force UTF-8 for EarthEngine subprocess
        ee_env['NO_COLOR'] = '1'
        ee_env['RICH_NO_COLOR'] = '1'
        ee_env['PY_COLORS'] = '0'
        ee_env['CLICOLOR'] = '0'
        
        mcp_servers['EarthEngine'] = {
            "command": sys.executable,  # Use venv Python
            "args": ee_args,
            "transport": "stdio",
            "env": ee_env  # Pass env to subprocess (backup)
        }
    else:
        print("EarthEngine MCP not enabled (GEE credentials not set)")
    
    # Conditionally add OSM MCP server if OSM_MCP_PATH is set
    osm_mcp_path = os.getenv('OSM_MCP_PATH')
    if osm_mcp_path:
        print(f"OSM MCP enabled: {osm_mcp_path}")
        osm_env = dict(os.environ)
        osm_env['PYTHONIOENCODING'] = 'utf-8'  # Force UTF-8 for OSM subprocess
        osm_env['NO_COLOR'] = '1'
        osm_env['RICH_NO_COLOR'] = '1'
        osm_env['PY_COLORS'] = '0'
        osm_env['CLICOLOR'] = '0'
        mcp_servers['grabosm-osm-mcp'] = {
            "command": "node",
            "args": [osm_mcp_path],
            "transport": "stdio",
            "env": osm_env  # Pass env to subprocess
        }
    else:
        print("OSM MCP not enabled (OSM_MCP_PATH not set)")
    
    return llm, mcp_servers


def create_agents_from_tools(llm, osm_tools, earthengine_tools, main_tools):
    """
    Factory function to create agents from tools.
    Used both in production and tests.
    
    Args:
        llm: Main ChatOpenAI instance
        osm_tools: List of OSM tools for Location Agent
        earthengine_tools: List of EarthEngine tools for Data Acquisition Agent
        main_tools: List of remaining tools for Main Agent
    
    Returns:
        tuple: (main_agent, analyze_question_agent, location_agent, data_acquisition_agent)
    """
    # Create Analyze Question Agent with gpt-4o-mini (cost-efficient for analysis)
    analyze_question_llm = ChatOpenAI(
        model="gpt-4o-mini",
        temperature=0,  # Low temperature for consistent analysis
        api_key=llm.openai_api_key,
        base_url=llm.openai_api_base,
        use_responses_api=True,
        use_previous_response_id=True
    )
    analyze_question_agent = AnalyzeQuestionAgent(llm=analyze_question_llm)
    
    # Create Location Agent with gpt-4o-mini
    location_llm = ChatOpenAI(
        model="gpt-4o-mini",
        temperature=0.2,  # Low temperature for focused search
        api_key=llm.openai_api_key,
        base_url=llm.openai_api_base,
        use_responses_api=True,
        use_previous_response_id=True
    )
    location_agent = LocationAgent(
        llm=location_llm,
        osm_tools=osm_tools,
        max_iterations=30  # Увеличено: каждый tool call = 2 шага (вызов + результат)
    )
    
    # Create Data Acquisition Agent with gpt-4o
    data_acquisition_llm = ChatOpenAI(
        model="gpt-4o",
        temperature=0.2,  # Slightly higher temperature for creative retry strategies
        api_key=llm.openai_api_key,
        base_url=llm.openai_api_base,
        use_responses_api=True,
        use_previous_response_id=True
    )
    data_acquisition_agent = DataAcquisitionAgent(
        llm=data_acquisition_llm,
        earthengine_tools=earthengine_tools,
        max_iterations=30  # Увеличено для retry стратегий (поиск разных коллекций, cloud thresholds, временные окна)
    )
    
    # Create Main Agent
    main_llm = llm
    if USE_DEV_DATASET:
        # Faster/cheaper main model for development benchmark stability.
        main_llm = ChatOpenAI(
            model="gpt-4o-mini",
            temperature=0.1,
            api_key=llm.openai_api_key,
            base_url=llm.openai_api_base,
            use_responses_api=True,
            use_previous_response_id=True,
            model_kwargs={"parallel_tool_calls": False}
        )
    main_agent = create_react_agent(main_llm, main_tools)
    
    return main_agent, analyze_question_agent, location_agent, data_acquisition_agent


async def create_langchain_agent(llm, mcp_servers):
    """Create LangChain ReAct agent with MCP tools (with optional multi-agent support)"""
    # Create MCP client
    client = MultiServerMCPClient(mcp_servers)
    
    try:
        # Get tools from all MCP servers
        tools = await client.get_tools()
        
        # Dev mode: keep full tool list (no filtering)
        if USE_DEV_DATASET:
            # import sys
            # sys.path.insert(0, os.path.abspath('.'))
            # from dev_tools import filter_dev_tools
            # tools = filter_dev_tools(tools)
            print(f"Dev mode: loaded {len(tools)} tools (no filtering)")
            print(f"Available tools: {[tool.name for tool in tools]}")
        else:
            print(f"Successfully loaded {len(tools)} tools from MCP servers")
            print(f"Available tools: {[tool.name for tool in tools]}")
        
        # If multi-agent mode is enabled, split tools
        if USE_MULTIAGENT:
            # Separate tools for specialized agents
            osm_tools_async = [t for t in tools if t.name in [
                "search_location",
                "reverse_geocode",
                "search_structured",
                "get_place_details",
                "search_pois",
                "search_pois_smart",
                "find_amenities_nearby",
                "get_osrm_route",
                "get_distance_matrix",
                "optimize_route",
                "map_match_gps",
                "calculate_isochrone",
                "search_highways_smart",
                "get_elements_in_bounds",
                "search_by_tags",
                "get_route_data",
                "execute_overpass_query",
                "snap_to_roads",
                "get_changeset",
                "search_changesets",
                "get_changeset_diff",
                "osmose_search_issues",
                "osmose_get_issue_details",
                "osmose_get_issues_by_country",
                "osmose_get_issues_by_user",
                "osmose_get_stats",
                "osmose_get_items",
                "get_tag_suggestions",
                "get_tag_stats",
                "validate_osm_tag"
            ]]
            earthengine_tools_async = [t for t in tools if t.name in ["list_collections", "search_images", "download_band", "download_bands", "get_image_metadata"]]
            main_tools = [t for t in tools if t not in osm_tools_async and t not in earthengine_tools_async]
            
            # Wrap async tools to make them sync-compatible
            from langchain.tools import StructuredTool
            import asyncio
            
            def create_sync_wrapper(async_tool):
                """Create a sync wrapper for async tool"""
                def sync_invoke(**kwargs):
                    # Create new event loop for this thread
                    try:
                        loop = asyncio.get_event_loop()
                        if loop.is_closed():
                            raise RuntimeError("Event loop is closed")
                    except RuntimeError:
                        # No event loop in current thread, create new one
                        loop = asyncio.new_event_loop()
                        asyncio.set_event_loop(loop)
                    
                    try:
                        # Run async tool
                        result = loop.run_until_complete(async_tool.ainvoke(kwargs))
                        return result
                    except Exception as e:
                        # If that fails, try with asyncio.run (creates fresh loop)
                        try:
                            return asyncio.run(async_tool.ainvoke(kwargs))
                        except:
                            raise e
                
                return StructuredTool(
                    name=async_tool.name,
                    description=async_tool.description,
                    func=sync_invoke,
                    args_schema=async_tool.args_schema if hasattr(async_tool, 'args_schema') else None
                )
            
            osm_tools = [create_sync_wrapper(t) for t in osm_tools_async]
            earthengine_tools = [create_sync_wrapper(t) for t in earthengine_tools_async]
            
            print(f"🗺️ Multi-agent mode enabled!")
            print(f"   OSM tools for Location Agent: {len(osm_tools)} (wrapped for sync)")
            print(f"   EarthEngine tools for Data Acquisition Agent: {len(earthengine_tools)} (wrapped for sync)")
            print(f"   Main tools for Main Agent: {len(main_tools)}")
            
            # Use factory function to create agents
            main_agent, analyze_question_agent, location_agent, data_acquisition_agent = create_agents_from_tools(
                llm, osm_tools, earthengine_tools, main_tools
            )
            
            return (main_agent, analyze_question_agent, location_agent, data_acquisition_agent), client
        else:
            # Standard single-agent mode
            agent = create_react_agent(llm, tools)
            return agent, client
            
    except Exception as e:
        print(f"Error creating agent: {e}")
        if hasattr(client, 'close'):
            await client.close()
        raise


def load_questions(test_json_path: str = 'benchmark/question.json'):
    """Load evaluation questions"""
    # Use development dataset if configured
    if USE_DEV_DATASET:
        test_json_path = 'benchmark/development_quest.json'
        print(f"Using development dataset: {test_json_path}")
    
    with open(test_json_path, 'r', encoding='utf-8') as f:
        test_json = json.load(f)

    out = []
    for _, (question_idx, question_info) in enumerate(test_json.items()):
        # Filter to test subset if configured
        if USE_DEV_DATASET and DEV_TEST_QUESTION_IDS and question_idx not in DEV_TEST_QUESTION_IDS:
            continue
        
        # Handle different dataset formats
        if USE_DEV_DATASET:
            # Development dataset has only one evaluation entry (Instruction Following)
            eval_entry = question_info['evaluation'][0]
            data = eval_entry.get('data', None)
            question_text = eval_entry['question']

            # Fix data path: development questions data is in qual/benchmark/data/
            if data and data.startswith('benchmark/data/development_question'):
                data = data.replace('benchmark/data/', 'qual/benchmark/data/')

            # If local data path is missing/nonexistent, force dynamic acquisition path.
            has_local_data = bool(data and os.path.exists(data))
            if has_local_data:
                data_prompt = f"\nData location: {data}\n"
            else:
                # Use explicit marker to avoid ambiguous empty-path parsing by AnalyzeQuestionAgent.
                data_prompt = "\nData location: NOT_PROVIDED\n"

            out.append({
                "question_id": question_idx,
                "auto": question_text,  # Use same question for both modes in dev dataset
                "instruct": question_text,
                "data": data_prompt,
                "choices": question_info.get('choices', None)
            })
        else:
            # Main dataset has two evaluation entries (AP and IF)
            AP_INDEX = 0 if question_info['evaluation'][0]['type'] == 'autonomous planning' else 1
            data = question_info['evaluation'][AP_INDEX].get('data', None)
            data = question_info['evaluation'][1 - AP_INDEX].get('data', None) if data is None else data

            if data is None:
                continue
            out.append({
                "question_id": question_idx,
                "auto": question_info['evaluation'][AP_INDEX]['question'],
                "instruct": question_info['evaluation'][1 - AP_INDEX]['question'],
                "data": data,
                "choices": question_info.get('choices', None)
            })

    return out


def _extract_text_from_content(content) -> str:
    """Извлечь текст из content (совместимо с use_responses_api=True)
    
    При use_responses_api=True формат: content = [{'type': 'text', 'text': '...'}]
    При обычном режиме: content = "..."
    """
    # Если content - список словарей (Responses API)
    if isinstance(content, list):
        text_parts = []
        for item in content:
            if isinstance(item, dict) and item.get('type') == 'text':
                text_parts.append(item.get('text', ''))
        return ''.join(text_parts)
    
    # Если content - строка (обычный режим)
    return str(content) if content else ""


def normalize_multiple_choice_answer(text) -> str:
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

    answer_match = re.search(r"<Answer>(.*?)</Answer>", normalized, flags=re.IGNORECASE | re.DOTALL)
    answer_text = answer_match.group(1).strip() if answer_match else normalized

    patterns = (
        r"^\s*([A-F])\s*$",
        r"\(\s*([A-F])\s*\)",
        r"(?:вариант|option|answer|ответ|выбор)\s*[:\-]?\s*([A-F])\b",
        r"^\s*([A-F])\s*[\.\):\-]",
        r"\b([A-F])\b",
    )
    for pattern in patterns:
        match = re.search(pattern, answer_text, flags=re.IGNORECASE)
        if match:
            return f"<Answer>{match.group(1).upper()}</Answer>"

    return raw


def extract_answer_from_response(response, multiple_choice: bool = True):
    """Extract final answer from agent response"""
    messages = response.get("messages", [])
    
    # Look for the final answer in the last assistant message
    for message in reversed(messages):
        if hasattr(message, 'type') and message.type == 'ai':
            content = _extract_text_from_content(message.content)
            if '<Answer>' in content and '</Answer>' in content:
                # Extract answer between tags
                start = content.find('<Answer>') + len('<Answer>')
                end = content.find('</Answer>')
                if not multiple_choice:
                    return content[start:end].strip()
                return normalize_multiple_choice_answer(content[start:end].strip())
            return content
    
    return "No answer found"


def parse_agent_messages_to_conversations(messages, agent_name):
    """
    Parse agent messages into conversation log format.
    
    Unified function to parse messages from any agent (location, data_acquisition, main)
    into the same structured format.
    
    Args:
        messages: List of LangChain messages from agent
        agent_name: Name of agent ("location", "data_acquisition", "main")
    
    Returns:
        List of conversation log entries
    """
    conversation_entries = []
    
    for message in messages:
        if hasattr(message, 'type'):
            if message.type == 'ai':
                # Assistant message - может содержать текст и/или tool calls
                assistant_content = []
                
                # Add text content if present
                content_text = _extract_text_from_content(message.content)
                if content_text and content_text.strip():
                    assistant_content.append({
                        "type": "text",
                        "content": content_text
                    })
                
                # Add tool calls if present
                if hasattr(message, 'additional_kwargs') and 'tool_calls' in message.additional_kwargs:
                    for tool_call in message.additional_kwargs['tool_calls']:
                        try:
                            arguments = json.loads(tool_call['function']['arguments']) if isinstance(tool_call['function']['arguments'], str) else tool_call['function']['arguments']
                        except:
                            arguments = tool_call['function']['arguments']
                        
                        assistant_content.append({
                            "name": tool_call['function']['name'],
                            "input": arguments
                        })
                
                # Only add if we have content
                if assistant_content:
                    conversation_entries.append({
                        "role": "assistant",
                        "agent": agent_name,
                        "content": assistant_content
                    })
            
            elif message.type == 'tool':
                # Tool result
                conversation_entries.append({
                    "role": "tool",
                    "agent": agent_name,
                    "name": message.name,
                    "content": str(message.content)
                })
    
    return conversation_entries


async def handle_question(agent, question, chat_log_path, llm=None):
    """Handle a single question with the LangChain agent (supports multi-agent mode)"""
    try:
        # Prepare query
        query = question['auto'] + question['data'] if autoplanning else \
            question['instruct'] + question['data']

        # Add choices if present
        has_choices = question['choices'] and len(question['choices']) > 0
        if has_choices:
            query += '\n'.join([''] + [
                '{}.{}'.format(chr(ord('A') + i), choice) 
                for i, choice in enumerate(question['choices'])
            ])

        # Customize system prompt based on question type
        if has_choices:
            # Multiple choice question - require specific format
            system_prompt = SYSTEM_PROMPT_MULTIPLE_CHOICE
        else:
            # Open-ended question - allow free-form answer
            system_prompt = SYSTEM_PROMPT_OPEN_ENDED
        
        print(f"\n--- Processing Question {question['question_id']} ---")
        print(f"Query: {query[:200]}...")
        
        # Multi-agent mode: use Supervisor
        if USE_MULTIAGENT and isinstance(agent, tuple) and llm:
            main_agent, analyze_question_agent, location_agent, data_acquisition_agent = agent
            question_type = "multiple_choice" if has_choices else "open_ended"
            
            # Create supervisor
            supervisor = MultiAgentSupervisor(
                llm=llm,
                analyze_question_agent=analyze_question_agent,
                location_agent=location_agent,
                data_acquisition_agent=data_acquisition_agent,
                main_agent_executor=main_agent,
                system_prompt=system_prompt
            )
            
            print(f"🎯 Using Multi-Agent Supervisor (Location Agent + Data Acquisition Agent + Main Agent)")
            
            # Run supervisor
            result = await supervisor.run(
                question=query,
                question_type=question_type,
                question_id=str(question.get("question_id"))
            )
            
            final_answer = result.get("answer", "")
            metadata = result.get("metadata", {})
            
            print(f"🗺️ Location search used: {metadata.get('location_search_used', False)}")
            print(f"📍 Location found: {metadata.get('location_found', False)}")
            print(f"📡 Data acquisition used: {metadata.get('data_acquisition_used', False)}")
            print(f"📊 Data acquisition successful: {metadata.get('data_acquisition_successful', False)}")
            print(f"📝 Total steps: {metadata.get('total_steps', 0)}")
            
            # Log multi-agent conversation
            # В multi-agent режиме логируем ТОЛЬКО main agent conversation (без supervisor деталей)
            
            # Извлекаем main agent messages из result
            main_agent_raw_messages = result.get("main_agent_messages", [])
            
            # Конвертируем в формат как в обычном режиме
            conversation_log = []
            
            # Добавляем user message (с location контекстом если есть)
            user_content = query
            if metadata.get('location_found'):
                location_result = result.get("location_result", {})
                coords = location_result.get("coordinates", {})
                user_content += f"\n\n[Location Agent found: {coords.get('display_name', 'N/A')} at lat={coords.get('lat')}, lon={coords.get('lon')}]"
            
            conversation_log.append({
                "role": "user",
                "content": user_content
            })
            
            # === ADD SUB-AGENTS MESSAGES TO CONVERSATIONS ===
            # Extract messages from supervisor state
            supervisor_messages = result.get("messages", [])
            
            # Find Analyze Question Agent result
            for msg in supervisor_messages:
                if isinstance(msg, dict) and msg.get("agent") == "analyze_question_agent":
                    action = msg.get("action")
                    if action == "analysis":
                        analysis_content = msg.get("content", {})
                        # Log analysis result as assistant message
                        conversation_log.append({
                            "role": "assistant",
                            "agent": "analyze_question",
                            "content": json.dumps(analysis_content, ensure_ascii=False, indent=2)
                        })
                        break  # Only process once
            
            # Find Location Agent result with ALL its internal messages
            for msg in reversed(supervisor_messages):
                if isinstance(msg, dict) and msg.get("agent") == "location_agent":
                    action = msg.get("action")
                    if action == "search":
                        location_result_data = msg.get("result", {})
                        # Extract ALL messages from Location Agent's ReAct execution
                        location_agent_messages = location_result_data.get("messages", [])
                        
                        # Parse and add each message from Location Agent (same format as Main Agent)
                        for loc_msg in location_agent_messages:
                            if hasattr(loc_msg, 'type'):
                                if loc_msg.type == 'ai':
                                    # Check if this is a tool call
                                    if hasattr(loc_msg, 'additional_kwargs') and 'tool_calls' in loc_msg.additional_kwargs:
                                        tool_calls_list = []
                                        for tc in loc_msg.additional_kwargs.get('tool_calls', []):
                                            func_info = tc.get('function', {})
                                            tool_name = func_info.get('name', 'unknown')
                                            tool_args_str = func_info.get('arguments', '{}')
                                            try:
                                                tool_input = json.loads(tool_args_str) if isinstance(tool_args_str, str) else tool_args_str
                                            except:
                                                tool_input = tool_args_str
                                            
                                            tool_calls_list.append({
                                                "name": tool_name,
                                                "input": tool_input
                                            })
                                        
                                        if tool_calls_list:
                                            conversation_log.append({
                                                "role": "assistant",
                                                "agent": "location",
                                                "content": tool_calls_list
                                            })
                                    else:
                                        # Regular AI message (thinking)
                                        content_text = _extract_text_from_content(loc_msg.content)
                                        if content_text and content_text.strip():
                                            conversation_log.append({
                                                "role": "assistant",
                                                "agent": "location",
                                                "content": content_text
                                            })
                                elif loc_msg.type == 'tool':
                                    # Tool result
                                    conversation_log.append({
                                        "role": "tool",
                                        "agent": "location",
                                        "name": loc_msg.name,
                                        "content": str(loc_msg.content)
                                    })
                        break  # Only process once
            
            # Find Data Acquisition Agent result with ALL its internal messages
            for msg in reversed(supervisor_messages):
                if isinstance(msg, dict) and msg.get("agent") == "data_acquisition_agent":
                    action = msg.get("action")
                    if action == "search":
                        data_result = msg.get("result", {})
                        # Extract ALL messages from Data Acquisition Agent's ReAct execution
                        data_agent_messages = data_result.get("messages", [])
                        
                        # Parse and add each message from Data Acquisition Agent (same format as Main Agent)
                        for data_msg in data_agent_messages:
                            if hasattr(data_msg, 'type'):
                                if data_msg.type == 'ai':
                                    # Check if this is a tool call
                                    if hasattr(data_msg, 'additional_kwargs') and 'tool_calls' in data_msg.additional_kwargs:
                                        tool_calls_list = []
                                        for tc in data_msg.additional_kwargs.get('tool_calls', []):
                                            func_info = tc.get('function', {})
                                            tool_name = func_info.get('name', 'unknown')
                                            tool_args_str = func_info.get('arguments', '{}')
                                            try:
                                                tool_input = json.loads(tool_args_str) if isinstance(tool_args_str, str) else tool_args_str
                                            except:
                                                tool_input = tool_args_str
                                            
                                            tool_calls_list.append({
                                                "name": tool_name,
                                                "input": tool_input
                                            })
                                        
                                        if tool_calls_list:
                                            conversation_log.append({
                                                "role": "assistant",
                                                "agent": "data_acquisition",
                                                "content": tool_calls_list
                                            })
                                    else:
                                        # Regular AI message (thinking)
                                        content_text = _extract_text_from_content(data_msg.content)
                                        if content_text and content_text.strip():
                                            conversation_log.append({
                                                "role": "assistant",
                                                "agent": "data_acquisition",
                                                "content": content_text
                                            })
                                elif data_msg.type == 'tool':
                                    # Tool result
                                    conversation_log.append({
                                        "role": "tool",
                                        "agent": "data_acquisition",
                                        "name": data_msg.name,
                                        "content": str(data_msg.content)
                                    })
                        break  # Only process once
            
            # Парсим main agent messages в тот же формат что и в single-agent режиме
            for message in main_agent_raw_messages:
                if hasattr(message, 'type'):
                    if message.type == 'ai':
                        # Assistant message
                        assistant_content = []
                        
                        content_text = _extract_text_from_content(message.content)
                        if content_text and content_text.strip():
                            assistant_content.append({
                                "type": "text",
                                "content": content_text
                            })
                        
                        if hasattr(message, 'additional_kwargs') and 'tool_calls' in message.additional_kwargs:
                            for tool_call in message.additional_kwargs['tool_calls']:
                                try:
                                    arguments = json.loads(tool_call['function']['arguments']) if isinstance(tool_call['function']['arguments'], str) else tool_call['function']['arguments']
                                except:
                                    arguments = tool_call['function']['arguments']
                                
                                assistant_content.append({
                                    "name": tool_call['function']['name'],
                                    "input": arguments
                                })
                        
                        if assistant_content:
                            conversation_log.append({
                                "role": "assistant",
                                "agent": "main",
                                "content": assistant_content
                            })
                    
                    elif message.type == 'tool':
                        # Tool result
                        conversation_log.append({
                            "role": "tool",
                            "name": message.name,
                            "content": [{
                                "output": [{
                                    "text": str(message.content),
                                }],
                            }]
                        })
            
            # Write as JSON object (will be part of JSON array)
            log_entry = {
                "question_index": question['question_id'],
                "timestamp": datetime.now().strftime('%Y-%m-%d %H:%M:%S,%f')[:-3],
                "conversations": conversation_log,
                "final_answer": final_answer,
                "metadata": {
                    "mode": "multi_agent",
                    "location_search_used": metadata.get('location_search_used', False),
                    "location_found": metadata.get('location_found', False),
                    "data_acquisition_used": metadata.get('data_acquisition_used', False),
                    "data_acquisition_successful": metadata.get('data_acquisition_successful', False),
                    "analysis_contract": metadata.get('analysis_contract'),
                    "answer_decision_check": metadata.get('answer_decision_check')
                }
            }
            
            # Write to log file manually
            log_file = temp_dir_path / "{}_{}_langchain.json".format(
                model_name, 'AP' if autoplanning else "IF"
            )
            with open(log_file, 'a', encoding='utf-8') as f:
                f.write(json.dumps(log_entry, ensure_ascii=False, indent=4))
                f.write(',\n')
            
            # Save to chat log
            save_chat_message(chat_log_path, {
                "name": "multiagent_result",
                "role": "assistant",
                "content": final_answer,
                "metadata": metadata
            })
            
            print(f"Final Answer: {final_answer}")
            return final_answer
        
        # Standard single-agent mode
        full_query = f"{system_prompt}\n\nQuestion: {query}"
        
        # Save user message to chat log
        user_message = {
            "name": "user",
            "role": "user", 
            "content": full_query,
            "metadata": {"question_id": question['question_id']}
        }
        save_chat_message(chat_log_path, user_message)
        
        # Invoke agent with configuration to prevent infinite loops
        response = await agent.ainvoke(
            {"messages": [HumanMessage(content=full_query)]},
            config={
                "recursion_limit": 50,  # Increase recursion limit
                "max_execution_time": 300,  # 5 minutes timeout
            }
        )
        
        # Extract final answer
        final_answer = extract_answer_from_response(response, multiple_choice=has_choices)
        
        # Convert response to AgentScope-compatible format for logging
        conversation_log = []
        
        for message in response.get("messages", []):
            if hasattr(message, 'type'):
                if message.type == 'human':
                    # User message
                    conversation_log.append({
                        "role": "user",
                        "content": message.content
                    })
                
                elif message.type == 'ai':
                    # Assistant message - handle both thinking and tool calls
                    assistant_content = []
                    
                    # First check if there's thinking content (text before tool calls)
                    content_text = _extract_text_from_content(message.content)
                    if content_text and content_text.strip():
                        assistant_content.append({
                            "type": "text",
                            "content": content_text
                        })
                    
                    # Then check for tool calls
                    if hasattr(message, 'additional_kwargs') and 'tool_calls' in message.additional_kwargs:
                        # Format tool calls in AgentScope style
                        for tool_call in message.additional_kwargs['tool_calls']:
                            try:
                                arguments = json.loads(tool_call['function']['arguments']) if isinstance(tool_call['function']['arguments'], str) else tool_call['function']['arguments']
                            except:
                                arguments = tool_call['function']['arguments']
                            
                            assistant_content.append({
                                "name": tool_call['function']['name'],
                                "input": arguments
                            })
                    
                    # Only add to log if there's actual content
                    if assistant_content:
                        conversation_log.append({
                            "role": "assistant",
                            "agent": "main",  # Указываем что это главный агент
                            "content": assistant_content
                        })
                
                elif message.type == 'tool':
                    # Tool result message in AgentScope format
                    conversation_log.append({
                        "role": "tool",
                        "name": message.name,
                        "content": [{
                            "output": [{
                                "text": str(message.content),
                            }],
                        }]
                    })
        
        # Save detailed messages to .chat file (AgentScope format)
        for message in response.get("messages", []):
            if hasattr(message, 'type'):
                if message.type == 'human':
                    # Skip user message for .chat as it's already saved
                    continue
        
                elif message.type == 'ai':
                    # Assistant message - handle both thinking and tool calls for .chat file
                    assistant_chat_content = []
                    
                    # First check if there's thinking content (text before tool calls)
                    content_text = _extract_text_from_content(message.content)
                    if content_text and content_text.strip():
                        assistant_chat_content.append({
                            "type": "text",
                            "text": content_text
                        })
                    
                    # Then check for tool calls
                    if hasattr(message, 'additional_kwargs') and 'tool_calls' in message.additional_kwargs:
                        # Format tool calls like AgentScope
                        for tool_call in message.additional_kwargs['tool_calls']:
                            try:
                                arguments = json.loads(tool_call['function']['arguments']) if isinstance(tool_call['function']['arguments'], str) else tool_call['function']['arguments']
                            except:
                                arguments = tool_call['function']['arguments']
                            
                            assistant_chat_content.append({
                                "type": "tool_use",
                                "id": tool_call['id'],
                                "name": tool_call['function']['name'],
                                "input": arguments
                            })
                    
                    # Save assistant message with both thinking and tool calls
                    if assistant_chat_content:
                        assistant_message = {
                            "name": question['question_id'],
                            "role": "assistant",
                            "content": assistant_chat_content,
                            "metadata": None
                        }
                        save_chat_message(chat_log_path, assistant_message)
                
                elif message.type == 'tool':
                    # Tool result message
                    tool_result_message = {
                        "name": "system",
                        "role": "system",
                        "content": [{
                            "type": "tool_result",
                            "id": getattr(message, 'tool_call_id', 'unknown'),
                            "output": [{"type": "text", "text": str(message.content), "annotations": None, "meta": None}],
                            "name": message.name
                        }],
                        "metadata": None
                    }
                    save_chat_message(chat_log_path, tool_result_message)
        
        # Log the conversation in the same format as original code
        # Write as JSON object (will be part of JSON array)
        log_entry = {
            "question_index": question['question_id'],
            "timestamp": datetime.now().strftime('%Y-%m-%d %H:%M:%S,%f')[:-3],
            "conversations": conversation_log,
            "final_answer": final_answer
        }
        
        # Write to log file manually
        log_file = temp_dir_path / "{}_{}_langchain.json".format(
            model_name, 'AP' if autoplanning else "IF"
        )
        with open(log_file, 'a', encoding='utf-8') as f:
            f.write(json.dumps(log_entry, ensure_ascii=False, indent=4))
            f.write(',\n')
        
        print(f"Final Answer: {final_answer}")
        return final_answer
        
    except Exception as e:
        import traceback
        error_msg = f"Error processing question {question['question_id']}: {e}"
        print(error_msg)
        print("Full traceback:")
        traceback.print_exc()
        
        # Save error to chat log
        error_message = {
            "name": "system",
            "role": "system",
            "content": [{"type": "text", "content": error_msg}],
            "metadata": {"error": True, "question_id": question['question_id']}
        }
        save_chat_message(chat_log_path, error_message)
        
        # Log error
        log_entry = {
            "question_index": question['question_id'],
            "timestamp": datetime.now().strftime('%Y-%m-%d %H:%M:%S,%f')[:-3],
            "conversations": [],
            "final_answer": error_msg
        }
        
        # Write to log file manually
        log_file = temp_dir_path / "{}_{}_langchain.json".format(
            model_name, 'AP' if autoplanning else "IF"
        )
        with open(log_file, 'a', encoding='utf-8') as f:
            f.write(json.dumps(log_entry, ensure_ascii=False, indent=4))
            f.write(',\n')
        
        return f"Error: {e}"


async def process_questions_batch(agent, questions, llm, temp_dir, model_name, autoplanning):
    """
    Process batch of questions and save results to JSON files.
    
    Extracted from main() for testability.
    
    Args:
        agent: LangChain agent or tuple of (main, location, data_acq) agents
        questions: List of question dicts
        llm: LLM instance (for multi-agent mode)
        temp_dir: Path to temp directory
        model_name: Model name for output file
        autoplanning: Whether autoplanning is used
    
    Returns:
        Tuple of (results_path, log_file_path)
    """
    # Chat log file (shared across all questions in handle_question)
    chat_log_path = temp_dir / "chat.log"
    
    # Process questions
    results = []
    for question in tqdm(questions, desc="Processing questions"):
        answer = await handle_question(agent, question, chat_log_path, llm=llm)
        results.append({
            "question_id": question['question_id'],
            "answer": answer
        })
        
        # Optional: Add delay between questions to avoid rate limiting
        await asyncio.sleep(1)
    
    # Save results summary
    results_path = temp_dir / "results_summary.json"
    with open(results_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=4)
    
    # Close JSON array in log file
    log_file = temp_dir / "{}_{}_langchain.json".format(
        model_name, 'AP' if autoplanning else "IF"
    )
    
    # Fix trailing comma and close array
    with open(log_file, 'r', encoding='utf-8') as f:
        content = f.read()
    
    if content.rstrip().endswith(','):
        content = content.rstrip()[:-1]
    
    with open(log_file, 'w', encoding='utf-8') as f:
        f.write(content)
        f.write('\n]\n')
    
    return results_path, log_file


async def main():
    """Main evaluation function"""
    global USE_MULTIAGENT, USE_DEV_DATASET, DEV_TEST_QUESTION_IDS, autoplanning
    
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="LangChain-based Earth Science Agent with Multi-Agent Support")
    parser.add_argument("--use-multiagent", action="store_true", help="Enable multi-agent architecture (Location Agent + Main Agent)")
    parser.add_argument("--question-ids", nargs="+", help="Specific question IDs to process")
    parser.add_argument("--dev-mode", action="store_true", help="Use development dataset and dev-tools only")
    parser.add_argument("--autonomous", action="store_true", help="Use autonomous planning mode")
    args = parser.parse_args()
    
    # Apply command line arguments
    if args.use_multiagent:
        USE_MULTIAGENT = True
        print("🗺️ Multi-agent mode enabled via command line")
    
    if args.dev_mode:
        USE_DEV_DATASET = True
        print("🔧 Dev mode enabled via command line")
    
    if args.question_ids:
        DEV_TEST_QUESTION_IDS = args.question_ids
        print(f"📝 Processing specific questions: {DEV_TEST_QUESTION_IDS}")
    
    if args.autonomous:
        autoplanning = True
        print("🤖 Autonomous planning mode enabled")
    
    print("Initializing LangChain-based Earth Science Agent...")
    
    # Initialize global parameters
    init_global_params()
    
    # Initialize chat logger
    chat_log_path = init_chat_logger()
    print(f"Chat log will be saved to: {chat_log_path}")
    
    # Initialize multi-agent loggers
    if USE_MULTIAGENT:
        # Location Agent logger
        location_logger = logging.getLogger("LocationAgent")
        location_logger.setLevel(logging.INFO)
        location_log_path = temp_dir_path / "location_agent.log"
        location_handler = logging.FileHandler(location_log_path, encoding='utf-8')
        location_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
        location_logger.addHandler(location_handler)
        
        # Data Acquisition Agent logger
        data_acq_logger = logging.getLogger("DataAcquisitionAgent")
        data_acq_logger.setLevel(logging.INFO)
        data_acq_log_path = temp_dir_path / "data_acquisition_agent.log"
        data_acq_handler = logging.FileHandler(data_acq_log_path, encoding='utf-8')
        data_acq_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
        data_acq_logger.addHandler(data_acq_handler)
        
        # Supervisor logger
        supervisor_logger = logging.getLogger("Supervisor")
        supervisor_logger.setLevel(logging.INFO)
        supervisor_log_path = temp_dir_path / "supervisor.log"
        supervisor_handler = logging.FileHandler(supervisor_log_path, encoding='utf-8')
        supervisor_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
        supervisor_logger.addHandler(supervisor_handler)
        
        print(f"Location Agent logs: {location_log_path}")
        print(f"Data Acquisition Agent logs: {data_acq_log_path}")
        print(f"Supervisor logs: {supervisor_log_path}")
    
    # Load configuration and create agent
    llm, mcp_servers = load_langchain_config()
    agent_or_tuple, client = await create_langchain_agent(llm, mcp_servers)
    
    # Handle both single-agent and multi-agent returns
    if USE_MULTIAGENT and isinstance(agent_or_tuple, tuple):
        agent = agent_or_tuple  # It's (main_agent, location_agent, data_acquisition_agent) tuple
        print(f"✅ Multi-agent system created: Main Agent + Location Agent + Data Acquisition Agent")
    else:
        agent = agent_or_tuple  # It's single agent
        print(f"✅ Single agent created")
    
    try:
        # Load questions
        questions = load_questions()
        print(f"Loaded {len(questions)} questions for evaluation")
        
        # Use extracted function to process questions
        results_path, log_file = await process_questions_batch(
            agent=agent,
            questions=questions,
            llm=llm,
            temp_dir=temp_dir_path,
            model_name=model_name,
            autoplanning=autoplanning
        )
        
        print(f"\n{'='*80}")
        print(f"✅ Evaluation completed!")
        print(f"📁 Results: {results_path}")
        print(f"📁 Logs: {temp_dir_path}")
        print(f"📁 Chat log: {log_file}")
        if USE_MULTIAGENT:
            print(f"🗺️ Multi-agent mode was used")
        print(f"{'='*80}")
        
    except Exception as e:
        print(f"Error in main evaluation: {e}")
        raise
    
    finally:
        # Clean up
        if hasattr(client, 'close'):
            await client.close()


if __name__ == "__main__":
    asyncio.run(main())
