import pytest

def pytest_addoption(parser):
    parser.addoption("--main_path", action="store", default="main path")

@pytest.fixture
def main_path(request):
    return request.config.getoption("--main_path")
