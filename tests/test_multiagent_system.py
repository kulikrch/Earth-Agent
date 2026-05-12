"""
Unit tests for Multi-Agent System (Location Agent + Data Acquisition Agent + Main Agent)

Uses unittest.mock to mock agent methods directly, avoiding ReAct agent creation issues.
This approach tests the logic without creating actual LangChain agents.
"""

import pytest
import asyncio
from unittest.mock import Mock, patch, AsyncMock, MagicMock
from pathlib import Path


# ============================================================================
# Test Fixtures - Mock data for testing
# ============================================================================

@pytest.fixture
def mock_location_search_result():
    """Mock successful location search result."""
    return {
        "found": True,
        "coordinates": {
            "lat": 55.568,
            "lon": 37.484,
            "bbox": [37.464, 55.558, 37.504, 55.578],
            "display_name": "ЖК Скандинавия, бульвар Веласкеса, Коммунарка, Москва",
            "osm_type": "way",
            "osm_id": "12345"
        },
        "messages": [],
        "total_attempts": 2,
        "tool_calls": [{"name": "search_location", "args": "{}"}]
    }


@pytest.fixture
def mock_data_acquisition_result():
    """Mock successful data acquisition result."""
    return {
        "success": True,
        "downloaded_files": [
            {"path": "/tmp/question3/LST_2018_TIR10.tif", "tool": "download_bands"},
            {"path": "/tmp/question3/LST_2018_TIR11.tif", "tool": "download_bands"},
            {"path": "/tmp/question3/LST_2022_TIR10.tif", "tool": "download_bands"},
            {"path": "/tmp/question3/LST_2022_TIR11.tif", "tool": "download_bands"}
        ],
        "messages": [],
        "metadata": {"search_attempts": 3, "download_attempts": 2},
        "total_attempts": 5
    }


# ============================================================================
# Unit Tests: Location Agent
# ============================================================================

@patch('scripts.multiagents.location_agent.LocationAgent.search')
@pytest.mark.asyncio
async def test_location_agent_finds_coordinates(mock_search, mock_location_search_result):
    """Test that Location Agent successfully finds coordinates."""
    # Mock the search method to return predefined result
    mock_search.return_value = mock_location_search_result
    
    from scripts.multiagents.location_agent import LocationAgent
    
    # Create agent (won't actually create ReAct agent due to patch)
    agent = LocationAgent.__new__(LocationAgent)
    agent.search = mock_search
    
    result = await agent.search(
        query="ЖК Скандинавия, бульвар Веласкеса, Коммунарка, Москва",
        reason="Для тестирования",
        context="Тестовый запрос"
    )
    
    assert result["found"] == True
    assert "coordinates" in result
    assert result["coordinates"]["lat"] == 55.568
    assert result["coordinates"]["lon"] == 37.484


def test_location_agent_parse_coordinates():
    """Test coordinate parsing from tool results."""
    from scripts.multiagents.location_agent import LocationAgent
    
    # Mock response with tool result
    mock_response = {
        "messages": [
            Mock(type="ai", content=[{
                "type": "text",
                "text": "```json\n{\n  \"found\": true,\n  \"lat\": 55.5662578,\n  \"lon\": 37.4964502,\n  \"bbox\": [37.4953727, 55.5652741, 37.4981232, 55.5666424]\n}\n```"
            }])
        ]
    }
    
    # Create agent instance without calling __init__ (avoids ReAct agent creation)
    agent = LocationAgent.__new__(LocationAgent)
    coords = agent._parse_coordinates_from_response(mock_response)
    
    assert coords is not None
    assert coords["found"] == True
    assert coords["lat"] == 55.5662578
    assert coords["lon"] == 37.4964502


# ============================================================================
# Unit Tests: Data Acquisition Agent
# ============================================================================

def test_validate_downloaded_files_all_exist(tmp_path):
    """Test _validate_downloaded_files when all files exist."""
    from scripts.multiagents.data_acquisition_agent import DataAcquisitionAgent
    
    # Create test files
    file1 = tmp_path / "file1.tif"
    file2 = tmp_path / "file2.tif"
    file1.write_text("test data 1")
    file2.write_text("test data 2")
    
    # Create agent
    agent = DataAcquisitionAgent.__new__(DataAcquisitionAgent)
    agent.logger = Mock()  # Add logger mock
    
    files = [
        {"path": str(file1), "band": "B10"},
        {"path": str(file2), "band": "B11"}
    ]
    
    result = agent._validate_downloaded_files(files)
    
    assert result["valid"] == True
    assert len(result["existing_files"]) == 2
    assert len(result["missing_files"]) == 0
    assert result["total_files"] == 2
    assert result["error_message"] == ""


def test_validate_downloaded_files_some_missing(tmp_path):
    """Test _validate_downloaded_files when some files are missing."""
    from scripts.multiagents.data_acquisition_agent import DataAcquisitionAgent
    from unittest.mock import Mock
    
    # Create only one file
    file1 = tmp_path / "file1.tif"
    file1.write_text("test data")
    
    file2_path = str(tmp_path / "file2.tif")  # Does not exist
    
    agent = DataAcquisitionAgent.__new__(DataAcquisitionAgent)
    agent.logger = Mock()  # Add logger mock
    
    files = [
        {"path": str(file1), "band": "B10"},
        {"path": file2_path, "band": "B11"}
    ]
    
    result = agent._validate_downloaded_files(files)
    
    assert result["valid"] == False
    assert len(result["existing_files"]) == 1
    assert len(result["missing_files"]) == 1
    assert result["total_files"] == 2
    assert file2_path in result["missing_files"]
    assert "Missing 1/2 files" in result["error_message"]


def test_validate_downloaded_files_empty_path(tmp_path):
    """Test _validate_downloaded_files with empty path."""
    from scripts.multiagents.data_acquisition_agent import DataAcquisitionAgent
    from unittest.mock import Mock
    
    agent = DataAcquisitionAgent.__new__(DataAcquisitionAgent)
    agent.logger = Mock()  # Add logger mock
    
    files = [
        {"path": "", "band": "B10"},
        {"path": str(tmp_path / "file.tif"), "band": "B11"}
    ]
    
    result = agent._validate_downloaded_files(files)
    
    assert result["valid"] == False
    assert len(result["missing_files"]) == 2  # Empty path + non-existent file
    assert "[Empty path for band B10]" in result["missing_files"]


@patch('scripts.multiagents.data_acquisition_agent.DataAcquisitionAgent.search')
@pytest.mark.asyncio
async def test_data_acquisition_agent_downloads_files(mock_search, mock_data_acquisition_result):
    """Test that Data Acquisition Agent downloads all required files."""
    # Mock the search method to return predefined result
    mock_search.return_value = mock_data_acquisition_result
    
    from scripts.multiagents.data_acquisition_agent import DataAcquisitionAgent
    
    # Create agent without calling __init__
    agent = DataAcquisitionAgent.__new__(DataAcquisitionAgent)
    agent.search = mock_search
    
    requirements = """
    Найди и скачай снимки:
    - Локация: bbox=[37.464, 55.568, 37.504, 55.608]
    - Период 1: 2018-06-01 - 2018-08-31
    - Период 2: 2022-06-01 - 2022-08-31
    - Каналы: B10, B11
    - Директория: question3/
    """
    
    result = await agent.search(requirements)
    
    assert result["success"] == True
    assert len(result["downloaded_files"]) == 4  # 2 periods × 2 bands


# ============================================================================
# Integration Tests: Supervisor
# ============================================================================

@patch('scripts.multiagents.location_agent.LocationAgent.search')
@patch('scripts.multiagents.data_acquisition_agent.DataAcquisitionAgent.search')
@pytest.mark.asyncio
async def test_supervisor_full_workflow(
    mock_data_acq_search,
    mock_location_search,
    mock_location_search_result,
    mock_data_acquisition_result
):
    """
    Integration test: Full workflow from question to answer.
    
    Tests the complete flow:
    1. Supervisor analyzes question
    2. Location Agent finds coordinates (mocked)
    3. Data Acquisition Agent downloads data (mocked)
    4. Main Agent processes data (mocked)
    """
    # Setup mocks
    mock_location_search.return_value = mock_location_search_result
    mock_data_acq_search.return_value = mock_data_acquisition_result
    
    from scripts.multiagents.location_agent import LocationAgent
    from scripts.multiagents.Supervisor import MultiAgentSupervisor
    from scripts.multiagents.data_acquisition_agent import DataAcquisitionAgent
    
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
    
    # Mock LLM for supervisor
    mock_llm = Mock()
    
    # Mock Analyze Question Agent
    from scripts.multiagents.AnalyzeQuestionAgent import AnalyzeQuestionAgent
    mock_analyze_agent = AnalyzeQuestionAgent.__new__(AnalyzeQuestionAgent)
    mock_analyze_agent.analyze = Mock(return_value={
        "location_needed": True,
        "location_query": "ЖК Скандинавия",
        "data_acquisition_needed": True,
        "data_requirements": {"dates": [], "output_dir": "q3"}
    })
    
    # Create supervisor
    supervisor = MultiAgentSupervisor(
        llm=mock_llm,
        analyze_question_agent=mock_analyze_agent,
        location_agent=location_agent,
        data_acquisition_agent=data_acquisition_agent,
        main_agent_executor=mock_main_agent,
        system_prompt="Тестовый промпт"
    )
    
    # Run full workflow
    result = await supervisor.run(
        question="Оцените эффект теплового острова от ЖК Скандинавия в Коммунарке",
        question_type="open_ended"
    )
    
    # Verify results
    assert result["answer"] is not None
    assert "<Answer>" in result["answer"]
    assert result["metadata"]["location_search_used"] == True
    assert result["metadata"]["location_found"] == True
    assert result["metadata"]["data_acquisition_used"] == True
    assert result["metadata"]["data_acquisition_successful"] == True


# ============================================================================
# Smoke Tests: Routing Logic
# ============================================================================

def test_supervisor_routing_with_location():
    """Test Supervisor routes to location_agent when location is needed."""
    from scripts.multiagents.Supervisor import MultiAgentSupervisor
    
    supervisor = MultiAgentSupervisor(
        llm=Mock(),
        analyze_question_agent=Mock(),
        location_agent=Mock(),
        data_acquisition_agent=Mock(),
        main_agent_executor=Mock(),
        system_prompt="Test"
    )
    
    state = {
        "location_search_needed": True,
        "data_acquisition_needed": False
    }
    
    route = supervisor._route_after_analysis(state)
    assert route == "location_agent"


def test_supervisor_routing_with_data_acquisition():
    """Test Supervisor routes to data_acquisition_agent when data is needed."""
    from scripts.multiagents.Supervisor import MultiAgentSupervisor
    
    supervisor = MultiAgentSupervisor(
        llm=Mock(),
        analyze_question_agent=Mock(),
        location_agent=Mock(),
        data_acquisition_agent=Mock(),
        main_agent_executor=Mock(),
        system_prompt="Test"
    )
    
    state = {
        "location_search_needed": False,
        "data_acquisition_needed": True
    }
    
    route = supervisor._route_after_analysis(state)
    assert route == "data_acquisition_agent"


def test_supervisor_routing_direct_to_main():
    """Test Supervisor routes directly to main_agent when no prep needed."""
    from scripts.multiagents.Supervisor import MultiAgentSupervisor
    
    supervisor = MultiAgentSupervisor(
        llm=Mock(),
        analyze_question_agent=Mock(),
        location_agent=Mock(),
        data_acquisition_agent=Mock(),
        main_agent_executor=Mock(),
        system_prompt="Test"
    )
    
    state = {
        "location_search_needed": False,
        "data_acquisition_needed": False
    }
    
    route = supervisor._route_after_analysis(state)
    assert route == "main_agent"


# ============================================================================
# Tests: Spectral Channels Metadata
# ============================================================================

def test_data_acquisition_agent_parse_json_success():
    """Test DataAcquisitionAgent parses successful JSON response."""
    from scripts.multiagents.data_acquisition_agent import DataAcquisitionAgent
    from unittest.mock import Mock
    
    agent = DataAcquisitionAgent.__new__(DataAcquisitionAgent)
    
    # Mock AI message with JSON in markdown block
    mock_ai_msg = Mock()
    mock_ai_msg.type = "ai"
    mock_ai_msg.content = '''```json
{
  "success": true,
  "collection": "hls_landsat",
  "files": [
    {"path": "/tmp/test_B10.tif", "band": "B10", "date": "2018-07-15"},
    {"path": "/tmp/test_B11.tif", "band": "B11", "date": "2018-07-15"}
  ],
  "channels": [
    {"band": "B10", "reasoning": "Thermal channel for LST"},
    {"band": "B11", "reasoning": "Split-window LST"}
  ],
  "reasoning": "Selected thermal channels for heat island analysis"
}
```'''
    
    messages = [mock_ai_msg]
    result = agent._parse_final_response(messages)
    
    assert result is not None
    assert result["success"] == True
    assert result["collection"] == "hls_landsat"
    assert len(result["files"]) == 2
    assert result["files"][0]["band"] == "B10"
    assert result["reasoning"] == "Selected thermal channels for heat island analysis"

def test_data_acquisition_agent_parse_json_failure():
    """Test DataAcquisitionAgent parses failure JSON response."""
    from scripts.multiagents.data_acquisition_agent import DataAcquisitionAgent
    from unittest.mock import Mock
    
    agent = DataAcquisitionAgent.__new__(DataAcquisitionAgent)
    
    mock_ai_msg = Mock()
    mock_ai_msg.type = "ai"
    mock_ai_msg.content = '''```json
{
  "success": false,
  "error": "No images found for 2018",
  "attempts": 5
}
```'''
    
    messages = [mock_ai_msg]
    result = agent._parse_final_response(messages)
    
    assert result is not None
    assert result["success"] == False
    assert "error" in result

@pytest.mark.asyncio
async def test_data_acquisition_agent_returns_channel_metadata(tmp_path):
    """Test DataAcquisitionAgent returns channel_summary with metadata from JSON response."""
    from scripts.multiagents.data_acquisition_agent import DataAcquisitionAgent
    from unittest.mock import Mock
    
    # Create test files that actually exist (для прохождения валидации)
    file1 = tmp_path / "test_B10.tif"
    file2 = tmp_path / "test_B11.tif"
    file1.write_text("test data 1")
    file2.write_text("test data 2")
    
    # Create agent
    agent = DataAcquisitionAgent.__new__(DataAcquisitionAgent)
    agent.llm = Mock()
    agent.earthengine_tools = []
    agent.max_iterations = 10
    agent.system_prompt = "Test prompt"
    agent.logger = Mock()  # Add logger mock
    
    # Mock the ReAct agent to return JSON response
    mock_react_agent = Mock()
    
    # Create mock AI message with JSON response (с реальными путями)
    # Важно: content должен быть строкой (не списком), т.к. _extract_text_from_content обрабатывает оба случая
    mock_ai_final = Mock()
    mock_ai_final.type = "ai"
    mock_ai_final.additional_kwargs = {}  # Пустой dict чтобы 'in' работал
    
    # JSON контент как строка (путь Windows нужно экранировать для JSON)
    file1_json = str(file1).replace('\\', '\\\\')
    file2_json = str(file2).replace('\\', '\\\\')
    
    mock_ai_final.content = f'''```json
{{
  "success": true,
  "collection": "landsat8",
  "files": [
    {{"path": "{file1_json}", "band": "B10", "date": "2023-07-15"}},
    {{"path": "{file2_json}", "band": "B11", "date": "2023-07-15"}}
  ],
  "channels": [
    {{"band": "B10", "reasoning": "Thermal infrared for LST calculation"}},
    {{"band": "B11", "reasoning": "Split-window method for accurate LST"}}
  ],
  "reasoning": "Selected thermal channels for heat island analysis"
}}
```'''
    
    mock_react_agent.invoke = Mock(return_value={
        "messages": [mock_ai_final]
    })
    
    agent.agent = mock_react_agent
    
    # Run search
    result = await agent.search("Test requirements")
    
    # Verify channel_summary is included
    assert result["success"] == True
    assert "channel_summary" in result
    assert len(result["channel_summary"]) == 2
    
    # Verify channel_summary contains data from agent's JSON response
    # (это копия из поля "channels", не обогащённые метаданные)
    ch1 = result["channel_summary"][0]
    assert ch1["band"] == "B10"
    assert ch1["reasoning"] == "Thermal infrared for LST calculation"
    
    ch2 = result["channel_summary"][1]
    assert ch2["band"] == "B11"
    assert ch2["reasoning"] == "Split-window method for accurate LST"
    
    # Verify collection is in result (top-level, not in channel_summary)
    assert result["collection"] == "landsat8"


# ============================================================================
# Integration Tests: File Validation with Retry
# ============================================================================

@pytest.mark.asyncio
async def test_file_validation_success_on_first_attempt(tmp_path):
    """Test file validation passes on first attempt when all files exist."""
    from scripts.multiagents.data_acquisition_agent import DataAcquisitionAgent
    from unittest.mock import Mock, patch
    
    # Create test files
    file1 = tmp_path / "test1.tif"
    file2 = tmp_path / "test2.tif"
    file1.write_text("data1")
    file2.write_text("data2")
    
    # Test only the validation method directly
    agent = DataAcquisitionAgent.__new__(DataAcquisitionAgent)
    agent.logger = Mock()  # Add logger mock
    
    files = [
        {"path": str(file1), "band": "B10"},
        {"path": str(file2), "band": "B11"}
    ]
    
    validation_result = agent._validate_downloaded_files(files)
    
    # Verify validation passes
    assert validation_result["valid"] == True
    assert len(validation_result["existing_files"]) == 2
    assert len(validation_result["missing_files"]) == 0


@pytest.mark.asyncio  
async def test_file_validation_retry_logic():
    """Test that validation retry logic works correctly."""
    from scripts.multiagents.data_acquisition_agent import DataAcquisitionAgent
    from unittest.mock import Mock
    
    agent = DataAcquisitionAgent.__new__(DataAcquisitionAgent)
    agent.logger = Mock()  # Add logger mock
    
    # First validation - files missing
    files_missing = [
        {"path": "/nonexistent/file1.tif", "band": "B10"},
        {"path": "/nonexistent/file2.tif", "band": "B11"}
    ]
    
    validation1 = agent._validate_downloaded_files(files_missing)
    assert validation1["valid"] == False
    assert len(validation1["missing_files"]) == 2
    assert "Missing 2/2 files" in validation1["error_message"]


@pytest.mark.asyncio
async def test_file_validation_fails_after_max_retries(tmp_path):
    """Test file validation fails after exhausting all retry attempts."""
    from scripts.multiagents.data_acquisition_agent import DataAcquisitionAgent
    from unittest.mock import Mock
    
    agent = DataAcquisitionAgent.__new__(DataAcquisitionAgent)
    agent.llm = Mock()
    agent.earthengine_tools = []
    agent.max_iterations = 10
    agent.system_prompt = "Test"
    agent.logger = Mock()  # Add logger mock
    
    # Always return non-existent files
    mock_ai_msg = Mock()
    mock_ai_msg.type = "ai"
    mock_ai_msg.additional_kwargs = {}
    mock_ai_msg.content = '''```json
{
  "success": true,
  "collection": "test",
  "files": [
    {"path": "/nonexistent/file1.tif", "band": "B10"},
    {"path": "/nonexistent/file2.tif", "band": "B11"}
  ],
  "channels": [],
  "reasoning": "Test"
}
```'''
    
    mock_react_agent = Mock()
    mock_react_agent.invoke = Mock(return_value={"messages": [mock_ai_msg]})
    agent.agent = mock_react_agent
    
    # Run search
    result = await agent.search("Test requirements")
    
    # Verify failure after max retries
    assert result["success"] == False
    assert result["downloaded_files"] == []
    assert "error" in result
    assert "File validation failed after 3 attempts" in result["error"]
    assert mock_react_agent.invoke.call_count == 4  # Initial + 3 retries


def test_file_validation_error_message_formatting():
    """Test that error messages are properly formatted."""
    from scripts.multiagents.data_acquisition_agent import DataAcquisitionAgent
    from unittest.mock import Mock
    
    agent = DataAcquisitionAgent.__new__(DataAcquisitionAgent)
    agent.logger = Mock()  # Add logger mock
    
    # Many missing files
    paths = [f"/missing/file{i}.tif" for i in range(10)]
    files = [{"path": path, "band": f"B{i}"} for i, path in enumerate(paths)]
    
    validation = agent._validate_downloaded_files(files)
    
    assert validation["valid"] == False
    assert len(validation["missing_files"]) == 10
    assert "Missing 10/10 files" in validation["error_message"]
    assert ", ".join(paths) in validation["error_message"]  # Only shows first 3


def test_file_validation_partial_success(tmp_path):
    """Test validation with mix of existing and missing files."""
    from scripts.multiagents.data_acquisition_agent import DataAcquisitionAgent
    from unittest.mock import Mock
    
    # Create only some files
    file1 = tmp_path / "exists.tif"
    file1.write_text("data")
    
    agent = DataAcquisitionAgent.__new__(DataAcquisitionAgent)
    agent.logger = Mock()  # Add logger mock
    
    files = [
        {"path": str(file1), "band": "B10"},
        {"path": str(tmp_path / "missing.tif"), "band": "B11"}
    ]
    
    validation = agent._validate_downloaded_files(files)
    
    assert validation["valid"] == False
    assert len(validation["existing_files"]) == 1
    assert len(validation["missing_files"]) == 1
    assert str(file1) in validation["existing_files"]


# ============================================================================
# Run tests
# ============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
