from devcon2026.cli import main


def test_cli_smoke() -> None:
    assert main([]) == 0
