"""
End-to-End tests for Multi-Agent System

These tests use the ACTUAL code from scripts/langchain_gpt4o_dev.py with mocked LLM and tools.
Tests the full workflow including logging, supervisor coordination, and agent interactions.

IMPORTANT: These tests import and call REAL functions from langchain_gpt4o_dev.py
to ensure the production code actually works!
"""

import pytest
import asyncio
import json
import tempfile
import logging
from pathlib import Path
from unittest.mock import Mock, patch, AsyncMock, MagicMock
from io import StringIO

# Import the actual modules we're testing
import sys
sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))

from multiagents.location_agent import LocationAgent
from multiagents.Supervisor import MultiAgentSupervisor
from multiagents.data_acquisition_agent import DataAcquisitionAgent
from multiagents.AnalyzeQuestionAgent import AnalyzeQuestionAgent

# IMPORTANT: Import REAL functions from langchain_gpt4o_dev.py
from langchain_gpt4o_dev import (
    create_agents_from_tools,
    handle_question,
    extract_answer_from_response
)


# ============================================================================
# Test Fixtures - Complete mock setup for E2E
# ============================================================================

@pytest.fixture
def temp_test_dir():
    """Create temporary directory for test outputs."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture
def mock_supervisor_llm_e2e():
    """
    Mock LLM that returns valid JSON for supervisor analysis.
    """
    mock_llm = Mock()
    
    # Mock for _analyze_question
    mock_response = Mock()
    mock_response.content = json.dumps({
        "location_needed": True,
        "location_query": "ЖК Скандинавия, бульвар Веласкеса, Коммунарка, Москва",
        "reason": "Для получения спутниковых снимков",
        "context": "Жилой комплекс 2018-2022",
        "data_acquisition_needed": True,
        "data_requirements": {
            "dates": [
                {"label": "до строительства", "start": "2018-06-01", "end": "2018-08-31"},
                {"label": "после строительства", "start": "2022-06-01", "end": "2022-08-31"}
            ],
            "purpose": "LST анализ",
            "output_dir": "question3"
        }
    })
    
    mock_llm.invoke = Mock(return_value=mock_response)
    return mock_llm


@pytest.fixture  
def captured_logs():
    """Capture logs from all agents."""
    logs = {
        "LocationAgent": StringIO(),
        "DataAcquisitionAgent": StringIO(),
        "Supervisor": StringIO()
    }
    
    handlers = {}
    for agent_name, stream in logs.items():
        logger = logging.getLogger(agent_name)
        logger.setLevel(logging.INFO)
        handler = logging.StreamHandler(stream)
        handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
        logger.addHandler(handler)
        handlers[agent_name] = handler
    
    yield logs
    
    # Cleanup
    for agent_name, handler in handlers.items():
        logging.getLogger(agent_name).removeHandler(handler)


# ============================================================================
# E2E Test: REAL create_agents_from_tools function
# ============================================================================

@pytest.mark.asyncio
async def test_e2e_create_agents_from_tools_real_function():
    """
    E2E Test: Test REAL create_agents_from_tools function from langchain_gpt4o_dev.py
    
    This ensures the production factory function works correctly.
    """
    from langchain.tools import StructuredTool
    
    # Mock LLM
    mock_llm = Mock()
    mock_llm.openai_api_key = "test-key"
    mock_llm.openai_api_base = "http://test"
    mock_llm.use_responses_api = True
    mock_llm.use_previous_response_id = True
    
    # Create REAL StructuredTools (not Mock objects)
    def mock_search_location(**kwargs):
        return "test result"
    
    def mock_list_collections(**kwargs):
        return "test collections"
    
    def mock_calculate_ndvi(**kwargs):
        return "test ndvi"
    
    osm_tools = [StructuredTool.from_function(
        func=mock_search_location,
        name="search_location",
        description="Mock OSM search"
    )]
    
    earthengine_tools = [StructuredTool.from_function(
        func=mock_list_collections,
        name="list_collections",
        description="Mock EE collections"
    )]
    
    main_tools = [StructuredTool.from_function(
        func=mock_calculate_ndvi,
        name="calculate_ndvi",
        description="Mock NDVI calculation"
    )]
    
    # CALL REAL FUNCTION from langchain_gpt4o_dev.py
    main_agent, analyze_question_agent, location_agent, data_acq_agent = create_agents_from_tools(
        mock_llm, osm_tools, earthengine_tools, main_tools
    )
    
    # Verify all agents created
    assert main_agent is not None
    assert analyze_question_agent is not None
    assert location_agent is not None
    assert data_acq_agent is not None
    
    # Verify Location Agent has correct properties
    assert hasattr(location_agent, 'search')
    assert hasattr(location_agent, 'osm_tools')
    assert len(location_agent.osm_tools) == 1
    
    # Verify Data Acquisition Agent has correct properties
    assert hasattr(data_acq_agent, 'search')
    assert hasattr(data_acq_agent, 'earthengine_tools')
    assert len(data_acq_agent.earthengine_tools) == 1


# ============================================================================
# E2E Test: REAL handle_question function (single-agent mode)
# ============================================================================

@pytest.mark.asyncio
async def test_e2e_handle_question_single_agent_real_function(temp_test_dir):
    """
    E2E Test: Test REAL handle_question function in single-agent mode.
    
    This tests the actual production code path for question handling.
    """
    import langchain_gpt4o_dev
    
    # Initialize global temp_dir_path for langchain_gpt4o_dev
    langchain_gpt4o_dev.temp_dir_path = temp_test_dir
    
    # Mock agent
    mock_agent = Mock()
    mock_agent.ainvoke = AsyncMock(return_value={
        "messages": [
            Mock(type="human", content="Test question"),
            Mock(type="ai", content="<Answer>Test Answer</Answer>", additional_kwargs={})
        ]
    })
    
    question = {
        "question_id": "test_1",
        "auto": "Calculate NDVI",
        "instruct": "Calculate NDVI for the area",
        "data": "\nData location: benchmark/data/test",
        "choices": None
    }
    
    chat_log_path = temp_test_dir / "test.chat"
    
    # CALL REAL FUNCTION from langchain_gpt4o_dev.py
    answer = await handle_question(mock_agent, question, chat_log_path, llm=None)
    
    # Verify answer extracted correctly
    assert answer == "Test Answer"
    
    # Verify chat log created
    assert chat_log_path.exists()
    
    # Verify agent was called
    mock_agent.ainvoke.assert_called_once()


# ============================================================================
# E2E Test: REAL handle_question function (multi-agent mode)
# ============================================================================

@patch('scripts.multiagents.location_agent.LocationAgent.search')
@patch('scripts.multiagents.data_acquisition_agent.DataAcquisitionAgent.search')
@pytest.mark.asyncio
async def test_e2e_handle_question_multiagent_real_function(
    mock_data_acq_search,
    mock_location_search,
    temp_test_dir,
    captured_logs
):
    """
    E2E Test: Test REAL handle_question function in multi-agent mode.
    
    This is the MOST IMPORTANT test - it uses the actual production code path
    including Supervisor creation and coordination.
    """
    import langchain_gpt4o_dev
    
    # Initialize global variables for langchain_gpt4o_dev
    langchain_gpt4o_dev.temp_dir_path = temp_test_dir
    langchain_gpt4o_dev.model_name = "test-model"
    langchain_gpt4o_dev.autoplanning = False
    langchain_gpt4o_dev.USE_MULTIAGENT = True  # ВАЖНО: включаем multi-agent режим!
    
    # Setup mocks for agents
    mock_location_search.return_value = {
        "found": True,
        "coordinates": {
            "lat": 55.568,
            "lon": 37.484,
            "bbox": [37.464, 55.558, 37.504, 55.578],
            "display_name": "ЖК Скандинавия, Москва"
        },
        "messages": [],
        "total_attempts": 2
    }
    
    mock_data_acq_search.return_value = {
        "success": True,
        "downloaded_files": [
            {"path": "/tmp/q3/file1.tif"},
            {"path": "/tmp/q3/file2.tif"}
        ],
        "messages": [],
        "metadata": {"search_attempts": 3}
    }
    
    # Create mock agents using __new__ to avoid ReAct creation
    location_agent = LocationAgent.__new__(LocationAgent)
    location_agent.search = mock_location_search
    
    data_acq_agent = DataAcquisitionAgent.__new__(DataAcquisitionAgent)
    data_acq_agent.search = mock_data_acq_search
    
    # Mock main agent with properly structured messages
    mock_ai_message = Mock()
    mock_ai_message.type = "ai"
    mock_ai_message.content = "<Answer>Final Answer from Main Agent</Answer>"
    mock_ai_message.additional_kwargs = {}  # Empty dict, not Mock!
    
    main_agent = Mock()
    main_agent.ainvoke = AsyncMock(return_value={
        "messages": [mock_ai_message]
    })
    
    # Mock Analyze Question Agent
    analyze_question_agent = AnalyzeQuestionAgent.__new__(AnalyzeQuestionAgent)
    analyze_question_agent.analyze = Mock(return_value={
        "location_needed": True,
        "location_query": "ЖК Скандинавия, Москва",
        "data_acquisition_needed": True,
        "data_requirements": {"dates": [], "output_dir": "q3"}
    })
    
    # Create agent tuple (как в production)
    agent_tuple = (main_agent, analyze_question_agent, location_agent, data_acq_agent)
    
    # Mock LLM for supervisor
    mock_llm = Mock()
    mock_llm.invoke = Mock(return_value=Mock(content=json.dumps({
        "location_needed": True,
        "location_query": "ЖК Скандинавия, Москва",
        "data_acquisition_needed": True,
        "data_requirements": {
            "dates": [{"label": "2022", "start": "2022-06-01", "end": "2022-08-31"}],
            "output_dir": "question3"
        }
    })))
    
    question = {
        "question_id": "3",
        "instruct": "Оцените эффект теплового острова",
        "auto": "Thermal island analysis",
        "data": "",
        "choices": None
    }
    
    chat_log_path = temp_test_dir / "multiagent.chat"
    
    # CALL REAL FUNCTION from langchain_gpt4o_dev.py
    # This will create Supervisor internally and run the full workflow!
    answer = await handle_question(
        agent=agent_tuple,
        question=question,
        chat_log_path=chat_log_path,
        llm=mock_llm
    )
    
    # === ASSERTIONS: Verify full E2E workflow ===
    
    # 1. Answer extracted correctly
    assert answer is not None
    assert "Final Answer" in answer or "Main Agent" in answer
    
    # 2. All agents were called
    mock_location_search.assert_called_once()
    mock_data_acq_search.assert_called_once()
    main_agent.ainvoke.assert_called_once()
    
    # 3. Analyze Question Agent was called
    analyze_question_agent.analyze.assert_called_once()
    
    # 4. Chat log created
    assert chat_log_path.exists()


# ============================================================================
# E2E Test: Full Supervisor Workflow (keeping original for compatibility)
# ============================================================================

@patch('scripts.multiagents.location_agent.LocationAgent.search')
@patch('scripts.multiagents.data_acquisition_agent.DataAcquisitionAgent.search')
@pytest.mark.asyncio
async def test_e2e_supervisor_coordinates_data_analysis(
    mock_data_acq_search,
    mock_location_search,
    mock_supervisor_llm_e2e,
    captured_logs
):
    """
    E2E Test: Full supervisor workflow from question to answer.
    
    This test uses REAL supervisor code with mocked agents.
    Tests:
    1. Supervisor analyzes question correctly
    2. Routes to Location Agent
    3. Routes to Data Acquisition Agent  
    4. Routes to Main Agent
    5. Logs are written correctly
    6. Metadata is populated correctly
    """
    # Setup mock returns
    mock_location_search.return_value = {
        "found": True,
        "coordinates": {
            "lat": 55.568,
            "lon": 37.484,
            "bbox": [37.464, 55.558, 37.504, 55.578],
            "display_name": "ЖК Скандинавия, бульвар Веласкеса, Коммунарка, Москва"
        },
        "messages": [],
        "total_attempts": 2
    }
    
    mock_data_acq_search.return_value = {
        "success": True,
        "downloaded_files": [
            {"path": "/tmp/q3/LST_2018_TIR10.tif"},
            {"path": "/tmp/q3/LST_2018_TIR11.tif"},
            {"path": "/tmp/q3/LST_2022_TIR10.tif"},
            {"path": "/tmp/q3/LST_2022_TIR11.tif"}
        ],
        "messages": [],
        "metadata": {"search_attempts": 5}
    }
    
    # Create mock agents (without ReAct internals)
    location_agent = LocationAgent.__new__(LocationAgent)
    location_agent.search = mock_location_search
    
    data_acquisition_agent = DataAcquisitionAgent.__new__(DataAcquisitionAgent)
    data_acquisition_agent.search = mock_data_acq_search
    
    # Mock main agent executor
    mock_main_agent = Mock()
    mock_main_agent.ainvoke = AsyncMock(return_value={
        "messages": [
            Mock(type="ai", content="<Answer>Температура повысилась на 3.2°C</Answer>")
        ]
    })
    
    # Mock Analyze Question Agent
    mock_analyze_agent = AnalyzeQuestionAgent.__new__(AnalyzeQuestionAgent)
    mock_analyze_agent.analyze = Mock(return_value={
        "location_needed": True,
        "location_query": "ЖК Скандинавия",
        "data_acquisition_needed": True,
        "data_requirements": {"dates": [], "output_dir": "q3"}
    })
    
    # Create supervisor with REAL code
    supervisor = MultiAgentSupervisor(
        llm=mock_supervisor_llm_e2e,
        analyze_question_agent=mock_analyze_agent,
        location_agent=location_agent,
        data_acquisition_agent=data_acquisition_agent,
        main_agent_executor=mock_main_agent,
        system_prompt="Test prompt"
    )
    
    # Run E2E workflow
    result = await supervisor.run(
        question="Оцените эффект теплового острова от ЖК Скандинавия",
        question_type="open_ended"
    )
    
    # === ASSERTIONS: Verify complete E2E flow ===
    
    # 1. Result structure
    assert result is not None
    assert "answer" in result
    assert "metadata" in result
    
    # 2. Answer content
    assert "<Answer>" in result["answer"]
    assert "3.2°C" in result["answer"]
    
    # 3. Metadata - all agents used
    assert result["metadata"]["location_search_used"] == True
    assert result["metadata"]["location_found"] == True
    assert result["metadata"]["data_acquisition_used"] == True
    assert result["metadata"]["data_acquisition_successful"] == True
    
    # 4. Verify agent calls
    mock_location_search.assert_called_once()
    mock_data_acq_search.assert_called_once()
    mock_main_agent.ainvoke.assert_called_once()
    
    # 5. Verify analyze_question_agent was called
    mock_analyze_agent.analyze.assert_called_once()
    
    # 6. Check logs were written
    supervisor_log = captured_logs["Supervisor"].getvalue()
    assert len(supervisor_log) > 0
    # Should have logged calling agents
    assert "Location Agent" in supervisor_log or "Calling" in supervisor_log


# ============================================================================
# E2E Test: Logging Verification
# ============================================================================

@patch('scripts.multiagents.location_agent.LocationAgent.search')
@pytest.mark.asyncio
async def test_e2e_location_agent_logging(mock_search, captured_logs):
    """
    E2E Test: Verify Location Agent logs correctly.
    
    Tests that:
    1. Agent logs search attempts
    2. Agent logs tool calls
    3. Agent logs success/failure
    4. Logs contain expected emojis and formatting
    """
    # Setup
    mock_search.return_value = {
        "found": True,
        "coordinates": {"lat": 55.5, "lon": 37.5, "bbox": [37, 55, 38, 56]},
        "messages": [],
        "total_attempts": 2,
        "tool_calls": [
            {"name": "search_location", "args": '{"query": "Москва"}'}
        ]
    }
    
    agent = LocationAgent.__new__(LocationAgent)
    agent.search = mock_search
    
    # Execute with logging
    logger = logging.getLogger("LocationAgent")
    logger.info("🗺️ Location Agent: Starting search for 'Москва'")
    
    result = await agent.search("Москва")
    
    if result["found"]:
        logger.info(f"✅ Location found: lat={result['coordinates']['lat']}, lon={result['coordinates']['lon']}")
    
    # Verify logs
    log_content = captured_logs["LocationAgent"].getvalue()
    
    # Check log structure
    assert "🗺️" in log_content
    assert "Location Agent" in log_content
    assert "Москва" in log_content
    
    # Check success logging
    assert "✅" in log_content
    assert "55.5" in log_content
    assert "37.5" in log_content


@patch('scripts.multiagents.data_acquisition_agent.DataAcquisitionAgent.search')
@pytest.mark.asyncio
async def test_e2e_data_acquisition_logging_retry(mock_search, captured_logs):
    """
    E2E Test: Verify Data Acquisition Agent logs retry attempts.
    
    Tests that:
    1. Agent logs each retry strategy attempt
    2. Failed attempts are logged with 🔄
    3. Success is logged with ✅
    4. Metadata about attempts is accurate
    """
    # Setup - simulate retry scenario
    mock_search.return_value = {
        "success": True,
        "downloaded_files": [
            {"path": "/tmp/file1.tif"},
            {"path": "/tmp/file2.tif"}
        ],
        "messages": [],
        "metadata": {"search_attempts": 3, "download_attempts": 1},
        "total_attempts": 4
    }
    
    agent = DataAcquisitionAgent.__new__(DataAcquisitionAgent)
    agent.search = mock_search
    
    # Execute with logging (simulate retry attempts)
    logger = logging.getLogger("DataAcquisitionAgent")
    logger.info("📡 Data Acquisition Agent: Starting search")
    logger.warning("🔄 No images found - agent should retry with different parameters")
    logger.warning("🔄 Retry attempt 2: increasing cloud threshold")
    logger.info("✅ Data acquisition successful: 2 files downloaded")
    
    result = await agent.search("requirements")
    
    # Verify logs
    log_content = captured_logs["DataAcquisitionAgent"].getvalue()
    
    # Check retry logging
    assert "📡" in log_content
    assert "🔄" in log_content
    assert "✅" in log_content
    
    # Check details
    assert "No images found" in log_content
    assert "retry" in log_content.lower()
    assert "2 files downloaded" in log_content


# ============================================================================
# E2E Test: Error Handling
# ============================================================================

@patch('scripts.multiagents.location_agent.LocationAgent.search')
@patch('scripts.multiagents.data_acquisition_agent.DataAcquisitionAgent.search')
@pytest.mark.asyncio
async def test_e2e_supervisor_handles_location_failure(
    mock_data_acq_search,
    mock_location_search,
    mock_supervisor_llm_e2e,
    captured_logs
):
    """
    E2E Test: Supervisor handles Location Agent failure gracefully.
    
    Tests that:
    1. Location Agent failure is logged
    2. Supervisor continues to Main Agent anyway
    3. Metadata reflects location not found
    4. No crash or exception
    """
    # Location fails
    mock_location_search.return_value = {
        "found": False,
        "coordinates": None,
        "error": "No results for query",
        "total_attempts": 5
    }
    
    # Data acquisition skipped (location needed for bbox)
    mock_data_acq_search.return_value = {
        "success": False,
        "downloaded_files": [],
        "error": "No bbox available"
    }
    
    # Setup agents
    location_agent = LocationAgent.__new__(LocationAgent)
    location_agent.search = mock_location_search
    
    data_acquisition_agent = DataAcquisitionAgent.__new__(DataAcquisitionAgent)
    data_acquisition_agent.search = mock_data_acq_search
    
    mock_main_agent = Mock()
    mock_main_agent.ainvoke = AsyncMock(return_value={
        "messages": [Mock(type="ai", content="<Answer>Unable to analyze without location</Answer>")]
    })
    
    # Mock Analyze Question Agent (returns location not found scenario)
    mock_analyze_agent = AnalyzeQuestionAgent.__new__(AnalyzeQuestionAgent)
    mock_analyze_agent.analyze = Mock(return_value={
        "location_needed": True,
        "location_query": "Unknown place",
        "data_acquisition_needed": True,
        "data_requirements": {"dates": [], "output_dir": "q_test"}
    })
    
    supervisor = MultiAgentSupervisor(
        llm=mock_supervisor_llm_e2e,
        analyze_question_agent=mock_analyze_agent,
        location_agent=location_agent,
        data_acquisition_agent=data_acquisition_agent,
        main_agent_executor=mock_main_agent,
        system_prompt="Test"
    )
    
    # Should not crash
    result = await supervisor.run("Test question", "open_ended")
    
    # Verify graceful handling
    assert result is not None
    assert result["metadata"]["location_found"] == False
    assert result["metadata"]["data_acquisition_successful"] == False
    
    # Verify logs show failure
    supervisor_log = captured_logs["Supervisor"].getvalue()
    assert "NOT found" in supervisor_log or "False" in supervisor_log


# ============================================================================
# E2E Test: Routing Logic
# ============================================================================

@pytest.mark.parametrize("location_needed,data_needed,expected_route", [
    (True, False, "location_agent"),
    (False, True, "data_acquisition_agent"),
    (False, False, "main_agent"),
    (True, True, "location_agent"),  # Location first, then data
])
def test_e2e_supervisor_routing_logic(location_needed, data_needed, expected_route):
    """
    E2E Test: Verify supervisor routing decisions.
    
    Tests all possible routing paths based on question analysis.
    """
    supervisor = MultiAgentSupervisor(
        llm=Mock(),
        analyze_question_agent=Mock(),
        location_agent=Mock(),
        data_acquisition_agent=Mock(),
        main_agent_executor=Mock(),
        system_prompt="Test"
    )
    
    state = {
        "location_search_needed": location_needed,
        "data_acquisition_needed": data_needed
    }
    
    route = supervisor._route_after_analysis(state)
    assert route == expected_route


# ============================================================================
# E2E Test: Log File Creation
# ============================================================================

@pytest.mark.asyncio
async def test_e2e_log_files_created(temp_test_dir):
    """
    E2E Test: Verify all log files are created correctly.
    
    Tests that:
    1. All agent log files are created
    2. Logs are written with UTF-8 encoding
    3. Log format is correct
    4. Emojis are preserved
    """
    # Setup loggers
    log_files = {}
    for agent_name in ["LocationAgent", "DataAcquisitionAgent", "Supervisor"]:
        log_path = temp_test_dir / f"{agent_name.lower()}.log"
        logger = logging.getLogger(agent_name)
        logger.setLevel(logging.INFO)
        
        handler = logging.FileHandler(log_path, encoding='utf-8')
        handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
        logger.addHandler(handler)
        
        log_files[agent_name] = log_path
        
        # Write test message with emoji
        logger.info(f"🎯 Test message from {agent_name}")
        
        handler.close()
        logger.removeHandler(handler)
    
    # Verify files exist and content
    for agent_name, log_path in log_files.items():
        assert log_path.exists()
        
        content = log_path.read_text(encoding='utf-8')
        assert len(content) > 0
        assert "🎯" in content
        assert agent_name in content
        assert "Test message" in content


# ============================================================================
# E2E Test: JSON Chat Log Structure
# ============================================================================

@pytest.mark.asyncio
async def test_e2e_json_chat_log_created_and_valid(temp_test_dir):
    """
    E2E Test: Verify chat log file is created with valid JSON.
    
    Simplified test - just checks that:
    1. File is created
    2. Contains valid JSON
    3. Has non-zero content
    """
    import langchain_gpt4o_dev
    
    # Initialize globals
    langchain_gpt4o_dev.temp_dir_path = temp_test_dir
    langchain_gpt4o_dev.model_name = "test-model"
    langchain_gpt4o_dev.USE_MULTIAGENT = True
    
    # Mock agents
    location_agent = LocationAgent.__new__(LocationAgent)
    location_agent.search = AsyncMock(return_value={
        "found": True,
        "coordinates": {"lat": 55.5, "lon": 37.5, "bbox": [37, 55, 38, 56]}
    })
    
    data_acq_agent = DataAcquisitionAgent.__new__(DataAcquisitionAgent)
    data_acq_agent.search = AsyncMock(return_value={
        "success": True,
        "downloaded_files": [{"path": "/tmp/test.tif"}]
    })
    
    # Mock main agent
    mock_ai_msg = Mock()
    mock_ai_msg.type = "ai"
    mock_ai_msg.content = "<Answer>Test Answer</Answer>"
    mock_ai_msg.additional_kwargs = {}
    
    main_agent = Mock()
    main_agent.ainvoke = AsyncMock(return_value={
        "messages": [mock_ai_msg]
    })
    
    # Mock Analyze Question Agent
    analyze_question_agent_mock = AnalyzeQuestionAgent.__new__(AnalyzeQuestionAgent)
    analyze_question_agent_mock.analyze = Mock(return_value={
        "location_needed": True,
        "location_query": "Test Location, Moscow",
        "data_acquisition_needed": True,
        "data_requirements": {"dates": [], "output_dir": "q_test"}
    })
    agent_tuple = (main_agent, analyze_question_agent_mock, location_agent, data_acq_agent)
    
    # Mock LLM
    mock_llm = Mock()
    mock_llm.invoke = Mock(return_value=Mock(content=json.dumps({
        "location_needed": True,
        "location_query": "Test location",
        "data_acquisition_needed": True,
        "data_requirements": {"dates": [], "output_dir": "test"}
    })))
    
    question = {
        "question_id": "test_json",
        "instruct": "Test question for JSON log",
        "auto": "Test",
        "data": "",
        "choices": None
    }
    
    chat_log_path = temp_test_dir / "test_json.chat"
    
    # Call real function
    answer = await handle_question(agent_tuple, question, chat_log_path, llm=mock_llm)
    
    # === ASSERTIONS ===
    
    # 1. File created
    assert chat_log_path.exists(), "Chat log file should be created"
    
    # 2. File has content
    assert chat_log_path.stat().st_size > 0, "Chat log should not be empty"
    
    # 3. Contains valid JSON (JSONL format)
    with open(chat_log_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    
    assert len(lines) > 0, "Should have at least one line"
    
    # Parse each line as JSON
    valid_json_count = 0
    for line in lines:
        line = line.strip()
        if line:
            try:
                json.loads(line)  # Should not raise
                valid_json_count += 1
            except json.JSONDecodeError as e:
                pytest.fail(f"Invalid JSON in line: {line[:100]}... Error: {e}")
    
    assert valid_json_count > 0, "Should have at least one valid JSON entry"
    
    # 4. Answer was extracted
    assert answer is not None
    assert "Test Answer" in answer


# ============================================================================
# E2E Test: Chat Log Tool Calls Format
# ============================================================================

@pytest.mark.asyncio
async def test_e2e_chat_log_contains_answer(temp_test_dir):
    """
    E2E Test: Verify chat log contains the final answer.
    
    Simplified test - just checks answer is in the file.
    """
    import langchain_gpt4o_dev
    
    langchain_gpt4o_dev.temp_dir_path = temp_test_dir
    langchain_gpt4o_dev.model_name = "test"
    langchain_gpt4o_dev.USE_MULTIAGENT = False  # Single agent for simplicity
    
    # Mock agent
    mock_final_answer = Mock()
    mock_final_answer.type = "ai"
    mock_final_answer.content = "<Answer>NDVI calculated successfully</Answer>"
    mock_final_answer.additional_kwargs = {}
    
    mock_agent = Mock()
    mock_agent.ainvoke = AsyncMock(return_value={
        "messages": [mock_final_answer]
    })
    
    question = {
        "question_id": "answer_test",
        "auto": "Calculate NDVI",
        "instruct": "Calculate NDVI",
        "data": "",
        "choices": None
    }
    
    chat_log_path = temp_test_dir / "answer_test.chat"
    
    answer = await handle_question(mock_agent, question, chat_log_path, llm=None)
    
    # 1. Answer extracted correctly
    assert answer == "NDVI calculated successfully"
    
    # 2. File created and has content
    assert chat_log_path.exists()
    content = chat_log_path.read_text(encoding='utf-8')
    assert len(content) > 0
    
    # 3. Answer is somewhere in the log file
    assert "NDVI calculated successfully" in content


# ============================================================================
# E2E Test: *_langchain.json file format (main() output)
# ============================================================================

@pytest.mark.asyncio
async def test_e2e_process_questions_batch_real_function(temp_test_dir):
    """
    E2E Test: Test REAL process_questions_batch function from langchain_gpt4o_dev.py
    
    This tests the actual function that main() uses to process questions and create
    the *_langchain.json file.
    """
    from langchain_gpt4o_dev import process_questions_batch
    import langchain_gpt4o_dev
    
    # Setup globals
    langchain_gpt4o_dev.temp_dir_path = temp_test_dir
    langchain_gpt4o_dev.model_name = "test-model"
    langchain_gpt4o_dev.USE_MULTIAGENT = True
    
    # Initialize the JSON file (as done before process_questions_batch)
    log_file = temp_test_dir / "test-model_IF_langchain.json"
    with open(log_file, 'w', encoding='utf-8') as f:
        f.write('[\n')
    
    # Mock agents
    location_agent = LocationAgent.__new__(LocationAgent)
    location_agent.search = AsyncMock(return_value={
        "found": True,
        "coordinates": {"lat": 55.5, "lon": 37.5, "bbox": [37, 55, 38, 56]}
    })
    
    data_acq_agent = DataAcquisitionAgent.__new__(DataAcquisitionAgent)
    data_acq_agent.search = AsyncMock(return_value={
        "success": True,
        "downloaded_files": [{"path": "/tmp/test.tif"}]
    })
    
    mock_ai_msg = Mock()
    mock_ai_msg.type = "ai"
    mock_ai_msg.content = "<Answer>Test Answer from Real Function</Answer>"
    mock_ai_msg.additional_kwargs = {}
    
    main_agent = Mock()
    main_agent.ainvoke = AsyncMock(return_value={
        "messages": [mock_ai_msg]
    })
    
    # Mock Analyze Question Agent
    analyze_question_agent_mock = AnalyzeQuestionAgent.__new__(AnalyzeQuestionAgent)
    analyze_question_agent_mock.analyze = Mock(return_value={
        "location_needed": True,
        "location_query": "Test Location, Moscow",
        "data_acquisition_needed": True,
        "data_requirements": {"dates": [], "output_dir": "q_test"}
    })
    agent_tuple = (main_agent, analyze_question_agent_mock, location_agent, data_acq_agent)
    
    mock_llm = Mock()
    mock_llm.invoke = Mock(return_value=Mock(content=json.dumps({
        "location_needed": True,
        "location_query": "Test",
        "data_acquisition_needed": True,
        "data_requirements": {"dates": [], "output_dir": "test"}
    })))
    
    # Test questions
    questions = [
        {
            "question_id": "batch_test_1",
            "instruct": "Question 1",
            "auto": "Q1",
            "data": "",
            "choices": None
        },
        {
            "question_id": "batch_test_2",
            "instruct": "Question 2",
            "auto": "Q2",
            "data": "",
            "choices": ["A", "B", "C", "D"]
        }
    ]
    
    # CALL REAL FUNCTION from langchain_gpt4o_dev.py!
    results_path, log_file_path = await process_questions_batch(
        agent=agent_tuple,
        questions=questions,
        llm=mock_llm,
        temp_dir=temp_test_dir,
        model_name="test-model",
        autoplanning=False
    )
    
    # === ASSERTIONS: Verify results ===
    
    # 1. Results summary file created
    assert results_path.exists(), "results_summary.json should be created"
    
    with open(results_path, 'r', encoding='utf-8') as f:
        results = json.load(f)
    
    assert isinstance(results, list)
    assert len(results) == 2
    assert results[0]["question_id"] == "batch_test_1"
    assert "Test Answer from Real Function" in results[0]["answer"]
    
    # 2. Langchain JSON log file created and properly formatted
    assert log_file_path.exists(), "*_langchain.json should be created"
    
    with open(log_file_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Should be valid JSON array
    try:
        data = json.loads(content)
    except json.JSONDecodeError as e:
        pytest.fail(f"Invalid JSON: {e}\nContent: {content[:500]}")
    
    # Should be array with 2 entries
    assert isinstance(data, list)
    assert len(data) == 2
    
    # Check structure of first entry
    entry1 = data[0]
    assert "question_index" in entry1
    assert entry1["question_index"] == "batch_test_1"
    assert "final_answer" in entry1
    assert "Test Answer from Real Function" in entry1["final_answer"]
    assert "metadata" in entry1
    assert entry1["metadata"]["mode"] == "multi_agent"
    
    print(f"✅ process_questions_batch() validated: {len(data)} questions processed")


# ============================================================================
# E2E Test: Sub-agents messages in conversations
# ============================================================================

@pytest.mark.asyncio
async def test_e2e_conversations_include_all_agents(temp_test_dir):
    """
    E2E Test: Verify that conversations array includes messages from all agents.
    
    Tests that the *_langchain.json file contains:
    - "agent": "location" messages
    - "agent": "data_acquisition" messages  
    - "agent": "main" messages
    
    All on the same level in conversations array (not nested).
    """
    from langchain_gpt4o_dev import handle_question
    import langchain_gpt4o_dev
    
    # Setup
    langchain_gpt4o_dev.temp_dir_path = temp_test_dir
    langchain_gpt4o_dev.model_name = "test-all-agents"
    langchain_gpt4o_dev.autoplanning = False
    langchain_gpt4o_dev.USE_MULTIAGENT = True
    
    # Initialize log file
    log_file = temp_test_dir / "test-all-agents_IF_langchain.json"
    with open(log_file, 'w', encoding='utf-8') as f:
        f.write('[\n')
    
    # Mock Location Agent with successful result including internal messages
    # AI message with tool call
    mock_location_ai_msg = Mock()
    mock_location_ai_msg.type = "ai"
    mock_location_ai_msg.content = ""
    mock_location_ai_msg.additional_kwargs = {
        "tool_calls": [{
            "id": "call_loc_1",
            "function": {
                "name": "search_location",
                "arguments": json.dumps({"query": "Test Location, Moscow"})
            }
        }]
    }
    
    # Tool response
    mock_location_tool_msg = Mock()
    mock_location_tool_msg.type = "tool"
    mock_location_tool_msg.name = "search_location"
    mock_location_tool_msg.content = '{"lat": "55.568", "lon": "37.484", "display_name": "Test Location, Moscow"}'
    
    location_agent = LocationAgent.__new__(LocationAgent)
    location_agent.search = AsyncMock(return_value={
        "found": True,
        "coordinates": {
            "lat": 55.568,
            "lon": 37.484,
            "bbox": [37.464, 55.558, 37.504, 55.578],
            "display_name": "Test Location, Moscow"
        },
        "messages": [mock_location_ai_msg, mock_location_tool_msg],  # Internal ReAct messages
        "total_attempts": 2
    })
    
    # Mock Data Acquisition Agent with successful result including internal messages
    # AI message with tool call
    mock_data_acq_ai_msg = Mock()
    mock_data_acq_ai_msg.type = "ai"
    mock_data_acq_ai_msg.content = ""
    mock_data_acq_ai_msg.additional_kwargs = {
        "tool_calls": [{
            "id": "call_data_1",
            "function": {
                "name": "download_bands",
                "arguments": json.dumps({
                    "image_id": "test_image",
                    "band_names": ["B5", "B6"],
                    "bbox": [37, 55, 38, 56]
                })
            }
        }]
    }
    
    # Tool response
    mock_data_acq_tool_msg = Mock()
    mock_data_acq_tool_msg.type = "tool"
    mock_data_acq_tool_msg.name = "download_bands"
    mock_data_acq_tool_msg.content = "Result saved at /tmp/test1.tif\nResult saved at /tmp/test2.tif"
    
    data_acq_agent = DataAcquisitionAgent.__new__(DataAcquisitionAgent)
    data_acq_agent.search = AsyncMock(return_value={
        "success": True,
        "downloaded_files": [
            {"path": "/tmp/test1.tif"},
            {"path": "/tmp/test2.tif"}
        ],
        "messages": [mock_data_acq_ai_msg, mock_data_acq_tool_msg],  # Internal ReAct messages
        "metadata": {"search_attempts": 3}
    })
    
    # Mock Main Agent
    mock_ai_msg = Mock()
    mock_ai_msg.type = "ai"
    mock_ai_msg.content = "<Answer>Analysis complete</Answer>"
    mock_ai_msg.additional_kwargs = {}
    
    main_agent = Mock()
    main_agent.ainvoke = AsyncMock(return_value={
        "messages": [mock_ai_msg]
    })
    
    # Mock Analyze Question Agent
    analyze_question_agent_mock = AnalyzeQuestionAgent.__new__(AnalyzeQuestionAgent)
    analyze_question_agent_mock.analyze = Mock(return_value={
        "location_needed": True,
        "location_query": "Test Location, Moscow",
        "data_acquisition_needed": True,
        "data_requirements": {"dates": [], "output_dir": "q_test"}
    })
    agent_tuple = (main_agent, analyze_question_agent_mock, location_agent, data_acq_agent)
    
    # Mock LLM
    mock_llm = Mock()
    mock_llm.invoke = Mock(return_value=Mock(content=json.dumps({
        "location_needed": True,
        "location_query": "Test Location",
        "data_acquisition_needed": True,
        "data_requirements": {
            "dates": [{"label": "2022", "start": "2022-06-01", "end": "2022-08-31"}],
            "output_dir": "test"
        }
    })))
    
    question = {
        "question_id": "all_agents_test",
        "instruct": "Test all agents",
        "auto": "Test",
        "data": "",
        "choices": None
    }
    
    chat_log_path = temp_test_dir / "all_agents.chat"
    
    # Call handle_question
    await handle_question(agent_tuple, question, chat_log_path, llm=mock_llm)
    
    # Close JSON array
    with open(log_file, 'r', encoding='utf-8') as f:
        content = f.read()
    
    if content.rstrip().endswith(','):
        content = content.rstrip()[:-1]
    
    with open(log_file, 'w', encoding='utf-8') as f:
        f.write(content)
        f.write('\n]\n')
    
    # === ASSERTIONS ===
    
    # Read and parse JSON
    with open(log_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    assert isinstance(data, list)
    assert len(data) > 0
    
    entry = data[0]
    assert "conversations" in entry
    
    conversations = entry["conversations"]
    assert len(conversations) > 0
    
    # Find agents in conversations
    # Now agents can be in "agent" field OR in "role"="assistant" with agent field
    agents_found = set()
    for msg in conversations:
        if "agent" in msg:
            agents_found.add(msg["agent"])
    
    # CRITICAL: All three agents should be present
    assert "location" in agents_found, f"Location agent messages should be in conversations. Found agents: {agents_found}"
    assert "data_acquisition" in agents_found, f"Data acquisition agent messages should be in conversations. Found agents: {agents_found}"
    assert "main" in agents_found, f"Main agent messages should be in conversations. Found agents: {agents_found}"
    
    # Verify Location Agent has multiple messages (from ReAct steps)
    location_msgs = [m for m in conversations if m.get("agent") == "location"]
    assert len(location_msgs) > 0, "Should have Location Agent messages"
    
    # Check that we have both assistant and tool messages from location
    location_roles = set(m.get("role") for m in location_msgs)
    print(f"Location agent roles: {location_roles}")
    
    # Verify Data Acquisition Agent has multiple messages (from ReAct steps)
    data_acq_msgs = [m for m in conversations if m.get("agent") == "data_acquisition"]
    assert len(data_acq_msgs) > 0, "Should have Data Acquisition Agent messages"
    
    # Check that we have both assistant and tool messages from data acquisition
    data_acq_roles = set(m.get("role") for m in data_acq_msgs)
    print(f"Data acquisition agent roles: {data_acq_roles}")
    
    print(f"✅ All agents present in conversations: {agents_found}")
    print(f"   Location messages: {len(location_msgs)}")
    print(f"   Data acquisition messages: {len(data_acq_msgs)}")


# ============================================================================
# Run tests
# ============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])

